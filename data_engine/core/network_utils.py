import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def get_robust_session(
    retries: int = 5, backoff_factor: float = 2.0, status_forcelist: list = None, timeout: tuple = (20, 60)
) -> requests.Session:
    """
    リトライロジックを組み込んだ堅牢な Session オブジェクトを返す。

    Args:
        retries (int): 最大リトライ回数
        backoff_factor (float): 指数バックオフの係数
        status_forcelist (list): リトライ対象のHTTPステータスコード
        timeout (tuple): (connect_timeout, read_timeout) デフォルト値

    Returns:
        requests.Session: 設定済みのセッション
    """
    if status_forcelist is None:
        # HF側の500(Internal Server Error)もリトライ対象として明示的に強化
        status_forcelist = [429, 500, 502, 503, 504]

    session = requests.Session()

    retry_strategy = Retry(
        total=retries,
        backoff_factor=backoff_factor,  # 2.0 (2, 4, 8, 16, 32s...)
        status_forcelist=status_forcelist,
        allowed_methods=["GET", "POST", "PUT"],  # 大規模コミット用のPUTも含める
        raise_on_status=False,
    )

    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    # 【極限強化】HF Hubの大規模コミット(300操作超)はサーバー側処理が重いため、Read Timeoutを180秒へ大幅延長
    original_request = session.request

    def robust_request(method, url, **kwargs):
        # httpx 互換の引数を requests 互換に変換
        if "follow_redirects" in kwargs:
            kwargs["allow_redirects"] = kwargs.pop("follow_redirects")

        if "timeout" not in kwargs:
            # (connect_timeout, read_timeout)
            # 【究極強化】HF Hubの大規模コミット(300操作超)はサーバー側処理が重いため、Read Timeoutを300秒へ大幅延長
            kwargs["timeout"] = (30, 300)
        return original_request(method, url, **kwargs)

    session.request = robust_request

    return session


# グローバルな堅牢セッションインスタンス (requests.Session がモンキーパッチされる前に作成)
GLOBAL_ROBUST_SESSION = get_robust_session()


def patch_all_networking():
    """
    プロジェクトで使用されているすべての主要な通信ライブラリに対し、
    urllib3ベースのリトライ戦略を備えたセッションを強制注入する。
    これこそが世界最高水準の安定性を実現する唯一の方法である。
    """
    from loguru import logger

    robust_session = GLOBAL_ROBUST_SESSION

    # 1. HuggingFace Hub の通信を堅牢化
    try:
        import huggingface_hub.utils._http as hf_http

        # 内部的な get_session を差し替える
        hf_http.get_session = lambda: robust_session
        logger.info("HF Hub communication has been robustified.")
    except ImportError:
        pass

    # 2. 外部ライブラリ parsing/edinet の通信を堅牢化
    modules_to_patch = []
    modules_to_patch.extend(
        [
            "data_engine.engines.parsing.edinet.edinet_api",
            "data_engine.engines.parsing.edinet.link_base_file_analyzer",
            "data_engine.engines.parsing.edinet.fs_tbl",
        ]
    )

    for mod_name in modules_to_patch:
        try:
            import sys

            if mod_name in sys.modules:
                mod = sys.modules[mod_name]
                if hasattr(mod, "requests"):
                    # Session クラス自体を差し替えて、インスタンス化時にアダプタ等が適用されるようにする
                    # (edinet_engine.py での実装と同様の考え方)
                    class RobustSessionAdapter:
                        def __init__(self, *args, **kwargs):
                            self._session = robust_session

                        def __getattr__(self, name):
                            return getattr(self._session, name)

                        def __enter__(self):
                            return self._session

                        def __exit__(self, *args):
                            pass

                    mod.requests.Session = RobustSessionAdapter

                    # 3. トップレベル関数の差し替え (直接呼び出し対策)
                    mod.requests.get = robust_session.get
                    mod.requests.post = robust_session.post
                    mod.requests.put = robust_session.put
                    mod.requests.delete = robust_session.delete
                    mod.requests.patch = robust_session.patch
                    mod.requests.head = robust_session.head
                    mod.requests.request = robust_session.request

                    logger.debug(f"Patched all networking entry points in {mod_name}")
        except Exception as e:
            logger.debug(f"Failed to patch {mod_name}: {e}")

    logger.debug("Network patching completed.")
