# All apis are aligned with MTIO Transformer so that other codes need no change

from typing import Literal

from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    PeftModel,
    PeftMixedModel
)
from torch import nn, Tensor
from transformers import (
    BertModel,
    BertConfig,
    GPT2Model,
    GPT2Config,
    PreTrainedModel
)
import torch


# from utils.common import mean_square_error
def mean_square_error(position_a: Tensor, position_b: Tensor, dimension: int = 2) -> Tensor:
    """
    Mean square error that considers the periodicity of viewport positions.
    """
    error = torch.abs(position_a - position_b)
    error = torch.minimum(error, torch.abs(position_a + 1 - position_b))
    error = torch.minimum(error, torch.abs(position_a - 1 - position_b))
    return torch.sum(error * error, dim=-1) / dimension


# CNN Feature Encoder in the NetLLM paper
class ViewportFeatureEncoder(nn.Module):
    '''
    input  size: (batch_size, timesteps, in_channels)
    output size: (batch_size, timesteps, out_channels)
    '''
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=out_channels, kernel_size=self.in_channels, stride=self.in_channels),
            nn.LeakyReLU()
        )

    def forward(self, x: Tensor) -> Tensor:
        batch_size, timesteps, _ = x.shape
        out = torch.empty((batch_size, 0, self.out_channels)).to(device=x.device)
        for t in range(timesteps):
            seq = x[:, t].unsqueeze(1)
            out = torch.cat((out, self.conv(seq).permute(0, 2, 1)), dim=1)
        return out


def peft_model(
    plm: PreTrainedModel,
    plm_type: Literal["bert", "gpt2"],
    rank: int,
    verbose: bool = False,
    task_type: TaskType = TaskType.FEATURE_EXTRACTION 
) -> PeftModel | PeftMixedModel:
    for param in plm.parameters():
        param.requires_grad = False
        if param.ndim == 1:
            param.data = param.data.to(torch.float32)
    plm.gradient_checkpointing_enable()
    plm.enable_input_require_grads()
    config = LoraConfig(
        r=rank,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type=task_type,
        target_modules=["query", "value"] if plm_type == "bert" else None,
        fan_in_fan_out=True if plm_type == "gpt2" else False
    )
    model = get_peft_model(plm, config)
    if verbose:

        def print_trainable_parameters(model: PeftModel | PeftMixedModel):
            trainable_params = 0
            all_param = 0
            for _, param in model.named_parameters():
                all_param += param.numel()
                if param.requires_grad:
                    trainable_params += param.numel()
            print(
                f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
            )
        print_trainable_parameters(model)

    return model


class ViewportPretrainedLM(nn.Module):
    def __init__(
        self,
        in_channels: int,
        fut_window : int,
        rank: int,
        conv_channels: int,
        plm_type: Literal["bert", "gpt2"]="gpt2",
        d_model: int = 768,
        n_layer: int = 12,
        device: str ='cuda:0',
    ) -> None:
        super().__init__()
        self.device = device
        self.in_channels = in_channels
        self.fut_window = fut_window
        self.d_model = d_model
        self.conv_channels = conv_channels
        self.feature_encoder = ViewportFeatureEncoder(in_channels=in_channels, out_channels=self.conv_channels).to(device=self.device)
        self.linear = nn.Linear(self.conv_channels,self.d_model).to(device=self.device)
        self.layernorm = nn.LayerNorm(normalized_shape=self.d_model).to(device=self.device)
        self.plm = peft_model(
            GPT2Model.from_pretrained(
                "openai-community/gpt2",
                config=GPT2Config(n_layer=n_layer, n_embd=self.d_model)
            ).to(device=self.device) if plm_type == "gpt2" else \
            BertModel.from_pretrained(
                "google-bert/bert-base-uncased",
                config=BertConfig(num_hidden_layers=n_layer, hidden_size=self.d_model)
            ).to(device=self.device),
            plm_type=plm_type,
            rank=rank,
            verbose=True,
        )
        self.net_head = nn.Sequential(nn.Linear(self.d_model, self.in_channels), nn.Sigmoid()).to(device=self.device)

    def forward(
        self, history: Tensor, current: Tensor, future: Tensor
    ) -> tuple[Tensor, Tensor]:
        """
        history shape (batch_size, hist_window, in_channels)
        current shape (batch_size, 1          , in_channels)
        future  shape (batch_size, fut_window , in_channels)
        """
        assert history.dim() == future.dim() == current.dim()
        assert history.dim() == 3
        hist_window: int = history.shape[1]
        tgt: Tensor = torch.cat((history, current), dim=1).to(device=self.device) # (batch_size, hist_window + 1, in_channels)
        for _ in range(self.fut_window):
            tgt_embed: Tensor = self.feature_encoder(tgt)
            tgt_embed = self.linear(tgt_embed)
            tgt_embed = self.layernorm(tgt_embed)
            out : Tensor = self.plm(inputs_embeds=tgt_embed).last_hidden_state   # (batch_size, i, d_model)
            pred: Tensor = self.net_head(out[:, -1].unsqueeze(1))
            tgt = torch.cat((tgt, pred), dim=1)
        pred = tgt[:, hist_window + 1:]
        return pred, future
    
    def loss_function(self, pred: Tensor, gt: Tensor) -> Tensor:
        return torch.mean(mean_square_error(pred, gt))
    
    def sample(self, history: Tensor, current: Tensor) -> Tensor:
        assert history.dim() == current.dim() == 3
        hist_window: int = history.shape[1]
        tgt: Tensor = torch.cat((history, current), dim=1).to(device=self.device) # (batch_size, hist_window + 1, in_channels)
        for _ in range(self.fut_window):
            tgt_embed: Tensor = self.feature_encoder(tgt)
            tgt_embed = self.linear(tgt_embed)
            tgt_embed = self.layernorm(tgt_embed)
            out : Tensor = self.plm(inputs_embeds=tgt_embed).last_hidden_state   # (batch_size, i, d_model)
            pred: Tensor = self.net_head(out[:, -1].unsqueeze(1))
            tgt = torch.cat((tgt, pred), dim=1)
        pred = tgt[:, hist_window + 1:]
        return pred


if __name__ == "__main__":
    # shape check
    hist  = torch.randn((128, 5, 2)).to(device="cuda:0")
    current = torch.randn((128, 1, 2)).to(device="cuda:0")
    label = torch.randn((128, 15, 2))
    model = ViewportPretrainedLM(in_channels=2, fut_window=15, rank=128, conv_channels=128, plm_type="gpt2")
    pred, fut = model(hist, current, label)
    print(pred.shape)
    print(fut.shape)