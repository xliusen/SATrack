import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttention(nn.Module):
    def __init__(self,device, dim, num_heads=4):
        super().__init__()
        self.device = device
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True).to(self.device)

    def forward(self, query_tokens, context_tokens):
        # query_tokens: [B, N_q, D]
        # context_tokens: [B, N_c, D]
        out, attn_weights = self.attn(query=query_tokens, key=context_tokens, value=context_tokens)
        return out, attn_weights  # attn_weights: [B, N_q, N_c]

class BidirectionalAttentionContrastiveLoss(nn.Module):
    def __init__(self, device, dim, temperature=0.07, top_k=3, num_heads=4, loss_weight=0.5):
        super().__init__()
        self.device = device
        self.temperature = temperature
        self.top_k = top_k
        self.cross_attn = CrossAttention(self.device, dim, num_heads=num_heads)
        self.loss_weight = loss_weight  # weighting between v->t and t->v

    # def compute_attention_score(self, query_tokens, context_tokens):
    #     _, attn_weights = self.cross_attn(query_tokens, context_tokens)  # [B, N_q, N_c]
    #     max_per_token, _ = attn_weights.max(dim=-1)  # [B, N_q]
    #     mean_score = max_per_token.mean(dim=-1)      # [B]
    #     return mean_score  # [B] scalar per sample
    def compute_attention_score(self, query_tokens, context_tokens):
        """
        query_tokens: [B, N_q, D]
        context_tokens: [B, N_c, D]
        Return: [B] attention-based semantic similarity scores
        """
        _, attn_weights = self.cross_attn(query_tokens, context_tokens)  # [B, N_q, N_c]

        # Step 1: Extract top-3 attention scores for each query token
        topk, _ = torch.topk(attn_weights, k=3, dim=-1)  # [B, N_q, 3]
        topk_mean_per_token = topk.mean(dim=-1)  # [B, N_q]

        # Step 2: Average over all query tokens
        score = topk_mean_per_token.mean(dim=-1)  # [B]

        # Step 3: Normalize or rescale
        range_val = score.max() - score.min()
        if range_val > 1e-4:
            score = (score - score.min()) / (range_val + 1e-6)
        else:
            score = score * 100  # fallback: direct rescale

        return score  # [B]

    def compute_directional_loss(self, anchor_tokens, target_tokens):
        """
        anchor_tokens: [B, N_a, D]  (e.g. vision)
        target_tokens: [B, N_t, D]  (e.g. language)
        """
        B = anchor_tokens.size(0)
        sim_matrix = torch.zeros(B, B, device=anchor_tokens.device)
        for i in range(B):
            for j in range(B):
                sim_matrix[i, j] = self.compute_attention_score(
                    anchor_tokens[i:i+1], target_tokens[j:j+1]
                )

        pos_sim = sim_matrix.diag().unsqueeze(1)  # [B, 1]
        sim_matrix.fill_diagonal_(-1e4)
        topk_neg_sim, _ = torch.topk(sim_matrix, k=min(self.top_k, B - 1), dim=1)  # [B, K]

        logits = torch.cat([pos_sim, topk_neg_sim], dim=1) / self.temperature  # [B, 1+K]
        labels = torch.zeros(B, dtype=torch.long, device=logits.device)
        loss = F.cross_entropy(logits, labels)
        return loss

    def forward(self, lang_tokens, vis_tokens):
        """
        lang_tokens: [B, N_l, D]
        vis_tokens: [B, N_v, D]
        """
        # vision -> language
        loss_v2t = self.compute_directional_loss(vis_tokens, lang_tokens)

        # language -> vision
        loss_t2v = self.compute_directional_loss(lang_tokens, vis_tokens)

        total_loss = self.loss_weight * loss_v2t + (1 - self.loss_weight) * loss_t2v
        return total_loss
