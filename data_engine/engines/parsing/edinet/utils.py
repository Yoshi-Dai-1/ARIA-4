import pandera as pa
import time
import contextlib
from loguru import logger



def get_columns_df(schema: pa.DataFrameModel) -> list:
    return list(schema.to_schema().columns.keys())


def get_dtype_dict(schema: pa.DataFrameModel) -> dict:
    return {name: col.dtype for name, col in schema.to_schema().columns.items()}


def remove_empty_lists(lst):
    return [x for x in lst if x]

def flatten_list(lst):
    flat_list = []
    for item in lst:
        if isinstance(item, list):
            flat_list.extend(flatten_list(item))
        else:
            flat_list.append(item)
    return flat_list

def format_taxonomi(taxonomi_str: str) -> str:
    """タクソノミ文字列のフォーマットを変換します。

    最後のアンダースコア('_')をコロン(':')に変換します。
    
    Args:
        taxonomi_str: 変換元のタクソノミ文字列（例: "jpcrp030000-asr_E37207-000_IncreaseDecreaseInIncomeTaxesPayableOpeCF"）
        
    Returns:
        変換後のタクソノミ文字列（例: "jpcrp030000-asr_E37207-000:IncreaseDecreaseInIncomeTaxesPayableOpeCF"）
        
    Raises:
        ValueError: 入力文字列が空またはNoneの場合
        ValueError: 入力文字列にアンダースコアが含まれていない場合
    
    Examples:
        >>> format_taxonomi("jpcrp030000-asr_E37207-000_IncreaseDecreaseInIncomeTaxesPayableOpeCF")
        "jpcrp030000-asr_E37207-000:IncreaseDecreaseInIncomeTaxesPayableOpeCF"
        >>> format_taxonomi("prefix_value")
        "prefix:value"
    """
    if not taxonomi_str:
        raise ValueError("入力文字列が空またはNoneです。有効なタクソノミ文字列を入力してください。")
    
    if '_' not in taxonomi_str:
        raise ValueError("入力文字列にアンダースコア('_')が含まれていません。適切なタクソノミ文字列を入力してください。")
    
    parts = taxonomi_str.split('_')
    return "_".join(parts[:-1]) + ":" + parts[-1]


@contextlib.contextmanager
def timer(name):
    t0=time.time()
    yield
    logger.info(f'[{name}] done in {time.time()-t0:.2f} s ')

