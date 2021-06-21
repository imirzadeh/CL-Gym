import torch
import torch.nn as nn
from typing import Iterable, Optional, Union, Iterable, Dict


class ContinualBackbone(nn.Module):
    """
    Base class for a continual backbone.
    Currently, this is a simple wrapper around PyTorch's `nn.Module` to support multiple heads.
    """
    def __init__(self,
                 multi_head: bool = False,
                 num_classes_per_head: Optional[int] = None):
        """
        Args:
            multi_head: Is this backbone multi-headed? Default: False.
            num_classes_per_head: If backbone is multi-headed, how many classes per head?
        """
        self.multi_head: bool = multi_head
        self.num_classes_per_head: int = num_classes_per_head
        if multi_head and not num_classes_per_head:
            raise ValueError("a Multi-Head Backbone is initiated without defining num_classes_per_head.")
        self.blocks: Union[Iterable[nn.Module], nn.ModuleList] = []
        super(ContinualBackbone, self).__init__()
    
    def get_block_params(self, block_id: int) -> Dict[str, torch.Tensor]:
        """
        Args:
            block_id: given the block number, provides the parameters.
        Returns:
            output: a dictionary of format {'param_name': params}
            
        . note::
            a block can have several layers (e.g., ResNet), or consist different parameters.
            For instance, the default `Linear` block has `
        """
        raise NotImplementedError
    
    @torch.no_grad()
    def get_block_outputs(self, inp: torch.Tensor, block_id: int, pre_act: bool = False) -> torch.Tensor:
        raise NotImplementedError
    
    def get_block_grads(self, block_id: int) -> torch.Tensor:
        raise NotImplementedError
     
    def select_output_head(self, output, head_ids: Iterable) -> torch.Tensor:
        """
        Helper method for selecting task-specific head.
        
        Args:
            output: The output of forward-pass. Shape: [BatchSize x ...]
            head_ids: head_ids for each example. Shape [BatchSize]

        Returns:
            output: The output where for each example in batch is calculated from one head in head_ids.
        """
        # TODO: improve performance by vectorizing this operation.
        # However, not too bad for now since number of classes is small (usually 2 or 5).
        for i, head in enumerate(head_ids):
            offset1 = int((head - 1) * self.num_classes_per_head)
            offset2 = int(head * self.num_classes_per_head)
            output[i, :offset1].data.fill_(-10e10)
            output[i, offset2:].data.fill_(-10e10)
        return output
    
    def forward(self, inp: torch.Tensor, head_ids: Optional[Iterable] = None) -> torch.Tensor:
        """
        Performs forward-pass
        
        Args:
            inp: The input tensor for forward-pass. size: [BatchSize x ...]
            head_ids: Optional list of classifier head ids. Size [BatchSize]

        Returns:
            output: Pytorch tensor of size [BatchSize x ...]
            
        . note::
            The `head_ids` will only be used if the backbone is multi-head.
        """
        out = inp
        for block in self.blocks:
            out = block(out)
        if self.multi_head:
            out = self.select_output_head(out, head_ids)
        return out
