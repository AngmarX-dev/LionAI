#!/usr/bin/env python3
"""Metadata-only hybrid proposal engine for LionAI; never touches file contents."""
from __future__ import annotations
import argparse
import json
import re
from collections.abc import Callable, Iterable
from pathlib import PurePosixPath
from typing import Any

SCHEMA_VERSION = "1.0"
MODEL_CONFIDENCE_THRESHOLD = 0.65

EXTENSION_GROUPS = {
    ".png":"Images",".jpg":"Images",".jpeg":"Images",".gif":"Images",".bmp":"Images",".webp":"Images",".tif":"Images",".tiff":"Images",".svg":"Images",".heic":"Images",
    ".pdf":"Documents",".doc":"Documents",".docx":"Documents",".odt":"Documents",".rtf":"Documents",".txt":"Documents",".md":"Documents",".ppt":"Documents",".pptx":"Documents",".xls":"Documents",".xlsx":"Documents",
    ".csv":"Data",".mp3":"Audio",".wav":"Audio",".flac":"Audio",".aac":"Audio",".ogg":"Audio",".m4a":"Audio",".wma":"Audio",
    ".mp4":"Video",".mkv":"Video",".mov":"Video",".avi":"Video",".webm":"Video",".wmv":"Video",".m4v":"Video",
    ".zip":"Archives",".7z":"Archives",".rar":"Archives",".tar":"Archives",".gz":"Archives",".bz2":"Archives",".xz":"Archives",
    ".c":"Source",".h":"Source",".cpp":"Source",".hpp":"Source",".cs":"Source",".java":"Source",".js":"Source",".ts":"Source",".py":"Source",".rs":"Source",".go":"Source",".html":"Source",".css":"Source",".php":"Source",".rb":"Source",".sh":"Source",
    ".json":"Config",".xml":"Config",".yml":"Config",".yaml":"Config",".toml":"Config",".ini":"Config",".env":"Config",
    ".exe":"Installers",".msi":"Installers",".msix":"Installers",".deb":"Installers",".rpm":"Installers",".apk":"Installers",".appimage":"Installers",
    ".ttf":"Fonts",".otf":"Fonts",".woff":"Fonts",".woff2":"Fonts",".db":"Databases",".sqlite":"Databases",".sqlite3":"Databases",".sql":"Databases",
}
KEYWORD_GROUPS = {
    "Reports":{"report","reports","invoice","receipt","statement","گزارش","فاکتور","رسید","صورت","حساب"},
    "Data":{"dataset","data","export","backup","records","داده","دیتاست","اطلاعات","خروجی"},
    "Source":{"source","src","code","script","main","program","کد","سورس","برنامه","اسکریپت"},
    "Config":{"config","settings","configuration","manifest","تنظیمات","پیکربندی","کانفیگ"},
    "Documents":{"notes","readme","manual","guide","letter","contract","یادداشت","راهنما","قرارداد","متن"},
    "Images":{"image","photo","picture","screenshot","cover","عکس","تصویر","پوستر"},
    "Audio":{"audio","music","song","track","podcast","موسیقی","آهنگ","صوت"},
    "Video":{"video","movie","clip","recording","ویدیو","فیلم","کلیپ"},
    "Archives":{"archive","compressed","package","آرشیو","فشرده"},
    "Databases":{"database","sqlite","db","دیتابیس","پایگاه"},
}

def _tokens(value: str) -> set[str]:
    return {x.casefold() for x in re.findall(r"[\w]+", value, flags=re.UNICODE)}

def _topic_folder(topic: str) -> str:
    chars = []
    for char in topic.strip():
        if char.isalnum(): chars.append(char)
        elif char == "\u200c": chars.append(char)
        elif char.isspace() or char in "-_.": chars.append("-")
    slug = re.sub(r"-+", "-", "".join(chars)).strip("-")[:48].strip("-")
    return slug or "Organized"

def _valid_filename(name: object) -> bool:
    return isinstance(name, str) and bool(name.strip()) and name not in {".",".."} and "/" not in name and "\\" not in name and "\x00" not in name

def _safe_relative_path(value: object, original_name: str) -> bool:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value: return False
    path = PurePosixPath(value)
    return (not path.is_absolute() and bool(path.parts)
            and all(part not in {"", ".", ".."} for part in path.parts)
            and path.name == original_name and ":" not in path.parts[0])

def _classify(name: str) -> tuple[str, float, str]:
    tokens = _tokens(PurePosixPath(name).stem)
    extension_group = EXTENSION_GROUPS.get(PurePosixPath(name).suffix.casefold())
    scores = {group: 2 * len(tokens & keywords) for group, keywords in KEYWORD_GROUPS.items() if tokens & keywords}
    best = max(scores, key=scores.get) if scores else None
    if best and scores[best] >= 2:
        return best, (0.82 if extension_group == best else 0.72), "filename_semantics"
    if extension_group: return extension_group, 0.76, "extension"
    return "Other/Review", 0.25, "uncertain"

ModelProposer = Callable[[str, list[dict[str, str]]], dict[str, Any]]

def _model_suggestions(topic: str, names: list[str], proposer: ModelProposer | None) -> dict[str, dict[str, Any]]:
    if proposer is None: return {}
    metadata = [{"name": n, "extension": PurePosixPath(n).suffix.casefold()} for n in names]
    response = proposer(topic, metadata)
    if not isinstance(response, dict) or not isinstance(response.get("files"), list): return {}
    accepted = {}
    for item in response["files"]:
        if not isinstance(item, dict): continue
        name, path = item.get("original_name"), item.get("suggested_path")
        confidence, review = item.get("confidence"), item.get("needs_review")
        if (isinstance(name, str) and name in names and name not in accepted and _safe_relative_path(path, name)
            and isinstance(confidence, (int,float)) and not isinstance(confidence,bool)
            and 0 <= confidence <= 1 and isinstance(review,bool) and not review
            and confidence >= MODEL_CONFIDENCE_THRESHOLD):
            accepted[name] = {"suggested_path":path,"confidence":float(confidence)}
    return accepted

def propose(topic: str, filenames: Iterable[str], model_proposer: ModelProposer | None = None) -> dict[str, Any]:
    """Produce a preview from topic + filename metadata; never read or move files."""
    if not isinstance(topic,str) or not topic.strip(): raise ValueError("topic must be a non-empty string")
    names = list(filenames)
    if not names or any(not _valid_filename(n) for n in names): raise ValueError("filenames must be plain filenames, not paths")
    if len(names) != len(set(names)): raise ValueError("filenames must be unique")
    root = _topic_folder(topic)
    model_map = _model_suggestions(topic,names,model_proposer)
    mappings, destinations = [], set()
    model_used = False
    for name in names:
        group, confidence, source = _classify(name)
        path = f"{root}/{group}/{name}"
        needs_review = group == "Other/Review"
        candidate = model_map.get(name)
        if candidate:
            path, confidence, source = candidate["suggested_path"], candidate["confidence"], "language_model"
            needs_review, model_used = False, True
        if path.casefold() in destinations:
            path, confidence, source, needs_review = f"{root}/Other/Review/{name}", 0.4, "collision_fallback", True
        destinations.add(path.casefold())
        mappings.append({"original_name":name,"suggested_path":path,"confidence":round(confidence,3),
                         "needs_review":needs_review,"decision_source":source})
    folders = sorted({str(PurePosixPath(x["suggested_path"]).parent) for x in mappings})
    return {"schema_version":SCHEMA_VERSION,"topic":topic.strip(),"engine":"hybrid-metadata-v1",
            "model_used":model_used,"folders":folders,"files":mappings,
            "safety":{"preview_only":True,"file_contents_read":False,"files_modified":False,
                      "requires_confirmation_to_apply":True}}

def main() -> int:
    parser=argparse.ArgumentParser(description="Create a metadata-only LionAI organization preview.")
    parser.add_argument("--topic",required=True)
    parser.add_argument("--files",nargs="+",required=True)
    args=parser.parse_args()
    print(json.dumps(propose(args.topic,args.files),ensure_ascii=False,indent=2))
    return 0

if __name__ == "__main__": raise SystemExit(main())
