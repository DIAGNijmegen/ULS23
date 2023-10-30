import os
from monai.data.folder_layout import FolderLayoutBase
from monai.config import PathLike


class FolderLayoutULS(FolderLayoutBase):
    def __init__(
        self,
        case_id: str,
        output_dir: PathLike,
        postfix: str = "",
        extension: str = "",
        use_idx: bool = True,
    ):
        self.case_id = case_id
        self.output_dir = output_dir
        self.postfix = postfix
        self.ext = extension
        self.use_idx = use_idx

    def filename(self, subject: PathLike = "subject", idx=None, **kwargs) -> PathLike:
        if self.use_idx:
            file_name = f"{self.output_dir}/{self.case_id}{self.postfix}{idx}{self.ext}"
        else:
            file_name = f"{self.output_dir}/{self.case_id}{self.postfix}{self.ext}"
        os.makedirs(self.output_dir, exist_ok=True)
        return file_name
