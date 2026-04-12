import requests
from huggingface_hub.utils import EntryNotFoundError, HfHubHTTPError, RepositoryNotFoundError
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def is_404(e: Exception) -> bool:
    """
    例外が HTTP 404 (Not Found) であるかどうかを厳密に判定する。
    
    ARIA の設計思想に基づき、ファイルが存在しない場合のみ True を返し、
    503 (Service Unavailable) や 504 (Gateway Timeout) などの通信障害時は
    False を延却することで、誤ったデータ初期化（サイレント消失）を防止する。
    """
    # 1. Hugging Face Hub 特有の "File Not Found" 例外
    if isinstance(e, (EntryNotFoundError, RepositoryNotFoundError)):
        return True
    
    # 2. Hugging Face Hub の HTTP エラー
    if isinstance(e, HfHubHTTPError):
        if e.response is not None and e.response.status_code == 404:
            return True
        # メッセージ文字列によるフォールバック検知
        if "404 Client Error" in str(e):
            return True
            
    # 3. 標準的な requests の HTTP エラー
    if isinstance(e, requests.exceptions.HTTPError):
        if e.response is not None and e.response.status_code == 404:
            return True

    return False


def get_robust_session(
    retries: int = 10, backoff_factor: float = 3.0, status_forcelist: list = None, timeout: tuple = (20, 60)
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
        # 429 (Rate Limit) と 500 系エラーを網羅。
        # 409 (Conflict) も稀に発生するため追加。
        status_forcelist = [429, 500, 502, 503, 504, 409]

    session = requests.Session()

    retry_strategy = Retry(
        total=retries,
        backoff_factor=backoff_factor,  # 3.0 (3, 9, 27, 81, 243s...)
        status_forcelist=status_forcelist,
        allowed_methods=["GET", "POST", "PUT", "HEAD"],  # HEAD を追加 (hf_hub_download等で使用)
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
        
        # 【重要】huggingface_hub の新版 (0.19+) は httpx を使用しており、
        # get_session が httpx.Client を返すことを期待している。
        # 旧来の requests.Session を渡すと .stream() メソッドがなくエラーになるため、
        # 必要に応じてアダプタを生成する。
        
        def get_compatible_hf_session():
            """httpx と requests のインターフェース差異を吸収するプロキシ"""
            session = GLOBAL_ROBUST_SESSION
            
            # もし get_session のアノテーションや実装が httpx を求めている場合
            if hasattr(hf_http, "httpx"):
                # 極めて簡易な httpx 互換プロキシ (duck typing)
                class HttpxCompatibilityProxy:
                    def __init__(self, requests_session):
                        self._s = requests_session
                    
                    def __getattr__(self, name):
                        return getattr(self._s, name)
                    
                    def request(self, method, url, **kwargs):
                        # httpx の follow_redirects を requests の allow_redirects に変換
                        if "follow_redirects" in kwargs:
                            kwargs["allow_redirects"] = kwargs.pop("follow_redirects")
                        return self._s.request(method, url, **kwargs)

                    from contextlib import contextmanager
                    @contextmanager
                    def stream(self, method, url, **kwargs):
                        # requests では get(..., stream=True) で対応
                        kwargs["stream"] = True
                        if "follow_redirects" in kwargs:
                            kwargs["allow_redirects"] = kwargs.pop("follow_redirects")
                        
                        resp = self._s.request(method, url, **kwargs)
                        
                        # Response オブジェクトにも httpx 互換のメソッドを生やす (Duck Typing)
                        if not hasattr(resp, "iter_bytes"):
                            resp.iter_bytes = resp.iter_content
                        
                        try:
                            yield resp
                        finally:
                            resp.close()

                return HttpxCompatibilityProxy(session)
            
            return session

        # 内部的な get_session を差し替える
        hf_http.get_session = get_compatible_hf_session
        logger.info("HF Hub communication has been robustified (Multi-Library Hybrid Patch).")
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
