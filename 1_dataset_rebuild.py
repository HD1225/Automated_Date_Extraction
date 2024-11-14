import pandas as pd
from pathlib import Path
import aiohttp
import asyncio
from tqdm.asyncio import tqdm_asyncio
from typing import Dict, List, Tuple
def clean_text(text):
    """
    remove space and not line
    """

    lines = text.splitlines()
    # 2. clean empty line and useless string
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        # replace with empty string
        line = ' '.join(line.split())
        if line:  # save only line
            cleaned_lines.append(line)

    # 3. make a string with each line
    cleaned_text = ' '.join(cleaned_lines)
    cleaned_text = f'"{cleaned_text}"'

    return cleaned_text
class DatasetProcessor:
    def __init__(self, min_length: int = 500):
        self.min_length = min_length
        self.valid_files: Dict[int, str] = {}
        self.failed_downloads: List[str] = []
        self.file_paths: Dict[str, str] = {}


    async def download_file(self, url: str, file_path: Path) -> Tuple[bool, str]:
        """异步下载单个文件"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        content = await response.text()
                        if len(content) >= self.min_length:
                            # 保存原始内容
                            file_path.write_text(content, encoding='utf-8')
                            # 返回带URL的内容和原始内容
                            content_with_url = f"{url}\n{content}"
                            return True, (content_with_url, content)
        except Exception as e:
            print(f"Error downloading {url}: {e}")
        return False, ("", "")

    async def process_dataset(self,
                              dataset: pd.DataFrame,
                              output_dir: Path) -> pd.DataFrame:
        """处理整个数据集"""
        output_dir.mkdir(exist_ok=True)

        # 准备下载任务
        tasks = []
        for idx, url in dataset['text version'].items():
            file_path = output_dir / f"dataset_pdf_{idx}.txt"
            tasks.append(self.download_file(url, file_path))

        # 并发下载
        results = await tqdm_asyncio.gather(*tasks)

        # 处理结果
        valid_mask = pd.Series(False, index=dataset.index)
        text_contents = {}
        raw_text_contents = {}

        for idx, (success, (content_with_url, raw_content)) in enumerate(results):
            if success:
                valid_mask.iloc[idx] = True
                content_with_url = clean_text(content_with_url)
                text_contents[idx] = content_with_url[:4000]
                raw_content = clean_text(raw_content)
                raw_text_contents[idx] = raw_content[:8000]

        # 更新数据集
        dataset_valid = dataset[valid_mask].copy()
        dataset_valid['local_filename'] = [
            f"dataset_pdf_{i}.txt" for i in dataset_valid.index
        ]
        dataset_valid['text_content'] = dataset_valid.index.map(text_contents)
        dataset_valid['raw_text_content'] = dataset_valid.index.map(raw_text_contents)

        return dataset_valid

async def main():
    # 配置
    dataset_path = Path("dataset.csv")
    output_dir = Path("./txt")

    # 加载数据
    data = pd.read_csv(dataset_path)

    # 处理数据
    processor = DatasetProcessor()
    dataset_valid = await processor.process_dataset(data, output_dir)

    # 保存结果
    dataset_valid.to_csv("dataset_valid.csv", index=False)
    print(f"Valid entries: {len(dataset_valid)}")
    print("\nSample with URL:")
    print(dataset_valid['text_content'].head(1))
    print("\nSample without URL:")
    print(dataset_valid['raw_text_content'].head(1))

if __name__ == "__main__":
    asyncio.run(main())