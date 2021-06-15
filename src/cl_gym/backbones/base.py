import torch
import torch.nn as nn
from typing import Iterable, Optional, Union, Iterable, Dict


class ContinualBackbone(nn.Module):
    def __init__(self,
                 multi_head: bool = False,
                 num_classes_per_head: Optional[int] = None):
        
        self.multi_head: bool = multi_head
        self.num_classes_per_head: int = num_classes_per_head
        self.blocks: Union[Iterable[nn.Module], nn.ModuleList] = []
        super(ContinualBackbone, self).__init__()
    
    def get_block_params(self, block_id: int) -> Dict[str, torch.Tensor]:
        raise NotImplementedError
    
    @torch.no_grad()
    def get_block_outputs(self, inp: torch.Tensor, block_id: int, pre_act: bool = False) -> torch.Tensor:
        raise NotImplementedError
    
    def get_block_grads(self, block_id: int) -> torch.Tensor:
        raise NotImplementedError
     
    def select_output_head(self, output, head_id: int):
        offset1 = int((head_id - 1) * self.num_classes_per_head)
        offset2 = int(head_id * self.num_classes_per_head)
        output[:, :offset1].data.fill_(-2e10)
        output[:, offset2:].data.fill_(-2e10)
        return output
    
    def forward(self, inp: torch.Tensor, head_id: Optional[int] = None) -> torch.Tensor:
        out = inp
        for block in self.blocks:
            out = block(out)
        if self.multi_head and head_id:
            out = self.select_output_head(out, head_id)
        return out
        
# if __name__ == "__main__":
