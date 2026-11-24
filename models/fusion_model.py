"""Gated Multimodal Fusion Architecture for Crisis Intelligence.

Fuses:
1. Social / Crisis Text Embeddings (e.g., BERTweet or DistilBERT).
2. Crisis Scene Image Embeddings (e.g., ConvNeXt-Tiny or ResNet-18).
Via:
- A Gated Multimodal Unit (GMU) with adaptive modality-weighting.
- Multi-task heads: Single-label Scenario Classification + Multi-label Humanitarian Needs.
- Native graceful fallback when one modality (image or text) is missing.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    nn = object  # type: ignore

try:
    from transformers import AutoModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False


if TORCH_AVAILABLE:
    class GatedMultimodalUnit(nn.Module):
        """Learns an adaptive gating scalar z in [0, 1] balancing text vs image representations.

        Formula:
            h_text = tanh(Linear(text_dim -> hidden_dim))
            h_vision = tanh(Linear(vision_dim -> hidden_dim))
            z = sigmoid(Linear([text_dim + vision_dim] -> hidden_dim))
            fused = z * h_text + (1 - z) * h_vision
        """

        def __init__(self, text_dim: int, vision_dim: int, hidden_dim: int = 512, dropout: float = 0.2):
            super().__init__()
            self.fc_text = nn.Linear(text_dim, hidden_dim)
            self.fc_vision = nn.Linear(vision_dim, hidden_dim)
            self.gate = nn.Linear(text_dim + vision_dim, hidden_dim)
            self.dropout = nn.Dropout(dropout)
            self.layer_norm = nn.LayerNorm(hidden_dim)

        def forward(
            self,
            text_feat: torch.Tensor,
            vision_feat: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            h_text = torch.tanh(self.fc_text(text_feat))
            h_vision = torch.tanh(self.fc_vision(vision_feat))

            # Compute modality importance gate
            combined = torch.cat([text_feat, vision_feat], dim=-1)
            gate_weights = torch.sigmoid(self.gate(combined))

            # Fused representation
            fused = (gate_weights * h_text) + ((1.0 - gate_weights) * h_vision)
            fused = self.layer_norm(self.dropout(fused))

            # Return fused vector and average gate score (near 1: text-dominated; near 0: vision-dominated)
            avg_gate_score = gate_weights.mean(dim=-1, keepdim=True)
            return fused, avg_gate_score


    class MultimodalCrisisClassifier(nn.Module):
        """End-to-End Multimodal Crisis Classifier with Multi-Task Prediction Heads."""

        def __init__(
            self,
            text_model_name: str = "vinai/bertweet-base",
            vision_model_name: str = "convnext_tiny",
            num_scenarios: int = 4,
            num_needs: int = 5,
            fusion_dim: int = 512,
            pretrained: bool = True,
            freeze_backbones: bool = False,
        ):
            super().__init__()
            self.text_model_name = text_model_name
            self.vision_model_name = vision_model_name
            self.fusion_dim = fusion_dim

            # 1. Text Encoder Setup
            text_dim = 768
            if TRANSFORMERS_AVAILABLE and pretrained:
                try:
                    self.text_encoder = AutoModel.from_pretrained(text_model_name)
                    text_dim = getattr(self.text_encoder.config, "hidden_size", 768)
                except Exception as e:
                    logger.warning("Could not load %s (%s). Using fallback DistilBERT.", text_model_name, e)
                    try:
                        self.text_encoder = AutoModel.from_pretrained("distilbert-base-uncased")
                        text_dim = 768
                    except Exception:
                        self.text_encoder = nn.Linear(96, 768)
            else:
                # Standalone linear projection for lightweight / offline initialization
                self.text_encoder = nn.Sequential(nn.Linear(96, 768), nn.ReLU())

            # 2. Vision Encoder Setup
            vision_dim = 768
            if TIMM_AVAILABLE and pretrained:
                try:
                    self.vision_encoder = timm.create_model(vision_model_name, pretrained=pretrained, num_classes=0)
                    vision_dim = self.vision_encoder.num_features
                except Exception as e:
                    logger.warning("Could not load %s (%s). Using fallback ResNet18.", vision_model_name, e)
                    self.vision_encoder = timm.create_model("resnet18", pretrained=pretrained, num_classes=0)
                    vision_dim = self.vision_encoder.num_features
            else:
                # Lightweight convolutional fallback
                self.vision_encoder = nn.Sequential(
                    nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(32, 768),
                )
                vision_dim = 768

            if freeze_backbones:
                for param in self.text_encoder.parameters():
                    param.requires_grad = False
                for param in self.vision_encoder.parameters():
                    param.requires_grad = False

            # 3. Gated Fusion Unit
            self.gmu = GatedMultimodalUnit(text_dim=text_dim, vision_dim=vision_dim, hidden_dim=fusion_dim)

            # Fallback projection layers for unimodal inference
            self.text_only_proj = nn.Sequential(
                nn.Linear(text_dim, fusion_dim),
                nn.ReLU(),
                nn.LayerNorm(fusion_dim),
            )
            self.vision_only_proj = nn.Sequential(
                nn.Linear(vision_dim, fusion_dim),
                nn.ReLU(),
                nn.LayerNorm(fusion_dim),
            )

            # 4. Multi-Task Heads
            # Head A: Disaster Scenario (Single-label Multi-class: Cyclone, Flood, Fire, None)
            self.scenario_head = nn.Sequential(
                nn.Linear(fusion_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, num_scenarios),
            )

            # Head B: Humanitarian Needs (Multi-label Binary: Rescue, Medical, Shelter, Food/Water, Evac)
            self.needs_head = nn.Sequential(
                nn.Linear(fusion_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, num_needs),
            )

        def extract_text_features(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
            if hasattr(self.text_encoder, "pooler_output") or hasattr(self.text_encoder, "config"):
                outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
                if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                    return outputs.pooler_output
                return outputs.last_hidden_state[:, 0, :]  # CLS token
            elif isinstance(self.text_encoder, nn.Module):
                return self.text_encoder(input_ids.float())
            return torch.zeros((input_ids.size(0), 768), device=input_ids.device)

        def extract_vision_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
            return self.vision_encoder(pixel_values)

        def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            pixel_values: Optional[torch.Tensor] = None,
        ) -> Dict[str, torch.Tensor]:
            """Forward pass with automatic modality detection (Multimodal, Text-only, or Vision-only)."""
            has_text = input_ids is not None
            has_vision = pixel_values is not None

            if has_text and has_vision:
                text_feat = self.extract_text_features(input_ids, attention_mask)
                vision_feat = self.extract_vision_features(pixel_values)
                fused, gate_score = self.gmu(text_feat, vision_feat)
                modality_mode = "multimodal"
            elif has_text:
                text_feat = self.extract_text_features(input_ids, attention_mask)
                fused = self.text_only_proj(text_feat)
                gate_score = torch.ones((text_feat.size(0), 1), device=text_feat.device)
                modality_mode = "text_only"
            elif has_vision:
                vision_feat = self.extract_vision_features(pixel_values)
                fused = self.vision_only_proj(vision_feat)
                gate_score = torch.zeros((vision_feat.size(0), 1), device=vision_feat.device)
                modality_mode = "vision_only"
            else:
                raise ValueError("Both text and image inputs are missing!")

            scenario_logits = self.scenario_head(fused)
            needs_logits = self.needs_head(fused)

            return {
                "scenario_logits": scenario_logits,
                "needs_logits": needs_logits,
                "fused_features": fused,
                "gate_score": gate_score,
                "modality_mode": modality_mode,
            }
else:
    # Dummy mock classes when torch is not installed
    class GatedMultimodalUnit:  # type: ignore
        def __init__(self, *args, **kwargs):
            pass

    class MultimodalCrisisClassifier:  # type: ignore
        def __init__(self, *args, **kwargs):
            pass
