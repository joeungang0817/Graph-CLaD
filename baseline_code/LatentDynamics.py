import copy
from einops import rearrange, repeat
import torch
import torch.nn as nn
import torch.nn.functional as F
from .Attentions import CrossAttnBlock
from .MlpResNet import FilmMLP, MlpResNet


def init_weight(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

def init_film_fc2_zero(m):
    if hasattr(m, 'cond_proj') and hasattr(m.cond_proj, 'fc2'):
        nn.init.zeros_(m.cond_proj.fc2.weight)
        nn.init.zeros_(m.cond_proj.fc2.bias)


class LatentDynamics(nn.Module):

    def __init__(self, proprio_dim, vl_dim, hidden_dim, action_dim):
        super().__init__()
        self.m_ema = 0.995
        self._ema_initialized = False

        self.type_emb_p = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.type_emb_s = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.type_emb_a = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

        self.p_query = nn.Parameter(torch.randn(1, 4, hidden_dim) * 0.02)
        self.s_query = nn.Parameter(torch.randn(1, 4, hidden_dim) * 0.02)
        self.dyn_query = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

        self.norm_p = nn.LayerNorm(hidden_dim)
        self.norm_s = nn.LayerNorm(hidden_dim)
        self.norm_a = nn.LayerNorm(hidden_dim)

        self.mask_token_p = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.mask_token_s = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

        self.p_backbone = MlpResNet(input_dim=proprio_dim, hidden_dim=hidden_dim, output_size=hidden_dim)
        self.s_backbone = FilmMLP(input_dim=vl_dim * 2, cond_dim=vl_dim, output_size=vl_dim)
        self.a_backbone = MlpResNet(input_dim=action_dim, hidden_dim=hidden_dim, output_size=hidden_dim)

        self.p_dyn = CrossAttnBlock(embed_dim=hidden_dim, dim_feedforward=hidden_dim, num_heads=4, num_layers=5, drop_out_rate=0.1)
        self.s_dyn = CrossAttnBlock(embed_dim=hidden_dim, dim_feedforward=hidden_dim, num_heads=4, num_layers=5, drop_out_rate=0.1)
        self.latent_dyn = CrossAttnBlock(embed_dim=hidden_dim, dim_feedforward=hidden_dim, num_heads=4, num_layers=5, drop_out_rate=0.1)
        self.dyn_readout = CrossAttnBlock(embed_dim=hidden_dim, dim_feedforward=hidden_dim, num_heads=4, num_layers=5, drop_out_rate=0.1)

        self.p_emb = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim), nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim)
            )
        self.s_emb = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim), nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim)
            )

        self.p_recon = nn.Sequential(nn.Mish(), nn.Linear(hidden_dim, proprio_dim))
        self.v_recon = nn.Sequential(nn.Mish(), nn.Linear(hidden_dim, vl_dim))

        self.apply(init_weight)
        self.apply(init_film_fc2_zero)

        self.p_backbone_target = copy.deepcopy(self.p_backbone)
        for p in self.p_backbone_target.parameters():
            p.requires_grad = False
        self.s_backbone_target = copy.deepcopy(self.s_backbone)
        for p in self.s_backbone_target.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update_ema(self):
        if not self._ema_initialized:
            self.p_backbone_target.load_state_dict(self.p_backbone.state_dict())
            self.s_backbone_target.load_state_dict(self.s_backbone.state_dict())
            self.p_backbone_target.eval()
            self.s_backbone_target.eval()
            self._ema_initialized = True
        self.momentum_update_target(self.p_backbone_target, self.p_backbone, self.m_ema)
        self.momentum_update_target(self.s_backbone_target, self.s_backbone, self.m_ema)

    @staticmethod
    def momentum_update_target(target_encoder, online_encoder, m):
        for param_q, param_k in zip(online_encoder.parameters(), target_encoder.parameters()):
            param_k.data.mul_(m).add_(param_q.data, alpha=1. - m)

    def _apply_token_mask(self, kv: torch.Tensor, mask_ratio: float, mask_token: torch.Tensor) -> torch.Tensor:
        if (not self.training) or (mask_ratio <= 0.0):
            return kv
        B, L, D = kv.shape
        mask = (torch.rand(B, L, device=kv.device) < mask_ratio)
        mask[:, 0] = False
        return torch.where(mask.unsqueeze(-1), mask_token.expand(B, L, D), kv)

    def forward(self, v_history, p_history, prev_action, lang, p_next=None, v_next=None, action_mask_ratio=0.3):
        B, V = v_history.shape[0], v_history.shape[2]

        p_prev = p_history[:, 0]  # [B, Dp]
        p_curr = p_history[:, -1]  # [B, Dp]
        v_prev = v_history[:, 0]  # [B, V, Dv]
        v_curr = v_history[:, -1]  # [B, V, Dv]

        p_prev = self.p_backbone(p_prev).unsqueeze(1) + self.p_query.expand(B, -1, -1)
        p_curr = self.p_backbone(p_curr).unsqueeze(1) + self.p_query.expand(B, -1, -1)

        v_prev = rearrange(v_prev, 'B V D -> B (V D)')
        v_curr = rearrange(v_curr, 'B V D -> B (V D)')
        s_prev = self.s_backbone(v_prev, lang).unsqueeze(1) + self.s_query.expand(B, -1, -1)
        s_curr = self.s_backbone(v_curr, lang).unsqueeze(1) + self.s_query.expand(B, -1, -1)

        p_prev = self.norm_p(p_prev) + self.type_emb_p
        p_curr = self.norm_p(p_curr) + self.type_emb_p

        s_prev = self.norm_s(s_prev) + self.type_emb_s
        s_curr = self.norm_s(s_curr) + self.type_emb_s

        action = self.a_backbone(prev_action)  # [B, H]
        action = self.norm_a(action).unsqueeze(1) + self.type_emb_a

        # Transition encoding
        p_sa = torch.cat([p_prev, action], dim=1)  # [B,2,D]
        p_sa = self._apply_token_mask(p_sa, action_mask_ratio, self.mask_token_p)
        s_sa = torch.cat([s_prev, action], dim=1)  # [B,2,D]
        s_sa = self._apply_token_mask(s_sa, action_mask_ratio, self.mask_token_s)
        latent_tp = self.p_dyn(p_curr, p_sa)  # [B,Np,D]
        latent_ts = self.s_dyn(s_curr, s_sa)  # [B,Ns,D]

        # Latent Dynamics
        latent_dyn = self.latent_dyn(latent_tp, latent_ts)  # [B,Np,D]

        # Learnable pooling
        latent_dyn = self.dyn_readout(self.dyn_query.expand(B, -1, -1), latent_dyn).squeeze(1)

        # JEPA-style prediction
        pred_p_emb = self.p_emb(latent_dyn)
        pred_p = self.p_recon(pred_p_emb)
        pred_s_emb = self.s_emb(latent_dyn)
        pred_v = self.v_recon(pred_s_emb)

        if self.training:
            # Prediction targets
            loss_dict = {}
            v_next_mv = v_next[:, 0]
            v_next_flat = rearrange(v_next, 'B V D -> B (V D)')
            with torch.no_grad():
                t_p_next = self.p_backbone_target(p_next)
                t_s_next = self.s_backbone_target(v_next_flat, lang)
                t_pn = F.normalize(t_p_next, dim=-1, eps=1e-8)
                t_sn = F.normalize(t_s_next , dim=-1, eps=1e-8)

            # Losses
            loss_p_emb = F.mse_loss(pred_p_emb, t_pn)
            loss_dict['loss_p'] = loss_p_emb

            loss_s_emb = F.mse_loss(pred_s_emb, t_sn)
            loss_dict['loss_s'] = loss_s_emb

            loss_p_recon = F.smooth_l1_loss(pred_p, p_next)
            loss_dict['loss_p_recon'] = loss_p_recon

            loss_v_recon = F.smooth_l1_loss(pred_v, v_next_mv)
            loss_dict['loss_v_recon'] = loss_v_recon

            return loss_dict

        return pred_p_emb, pred_s_emb
