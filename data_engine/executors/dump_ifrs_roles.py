import os
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import arelle.CntlrCmdLine
from arelle import ModelXbrl, ModelDocument

def dump_roles(zip_path: str):
    print(f"Inspecting {zip_path}...")
    with TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        with ZipFile(zip_path) as zf:
            xbrl_file = None
            for member in zf.namelist():
                if "PublicDoc" in member or "AuditDoc" in member:
                    zf.extract(member, temp_path)
                    if member.endswith(".xbrl") and "PublicDoc" in member:
                        xbrl_file = temp_path / member
        
        if not xbrl_file:
            print("No XBRL file found.")
            return

        from arelle import Cntlr
        cntlr = Cntlr.Cntlr()
        cntlr.startLogging(logFileName="logToPrint")
        model_xbrl = cntlr.modelManager.load(str(xbrl_file))
        
        print("\n=== Presentation Linkbase Roles ===")
        for role_uri, role_objs in model_xbrl.roleTypes.items():
            for role_obj in role_objs:
                def_str = str(role_obj.definition).lower()
                if "statement" in def_str or "position" in def_str or "income" in def_str or "cash" in def_str or "equity" in def_str:
                    print(f"- {role_uri}  ->  {role_obj.definition}")

if __name__ == "__main__":
    dump_roles("/Users/yoshi_dai/repos/ARIA/data/raw/edinet/year=2016/month=03/day=30/zip/S10079PD.zip")
