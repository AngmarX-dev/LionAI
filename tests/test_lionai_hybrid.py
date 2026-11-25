import unittest
from lionai_hybrid import propose

class HybridEngineTests(unittest.TestCase):
    def test_extension_and_semantic_classification(self):
        result=propose("Project files",["main.py","app_config.yaml","annual_report.pdf","holiday.jpg"])
        paths={x["original_name"]:x["suggested_path"] for x in result["files"]}
        self.assertIn("/Source/main.py",paths["main.py"])
        self.assertIn("/Config/app_config.yaml",paths["app_config.yaml"])
        self.assertIn("/Reports/annual_report.pdf",paths["annual_report.pdf"])
        self.assertIn("/Images/holiday.jpg",paths["holiday.jpg"])

    def test_unknown_file_requires_review(self):
        item=propose("Misc",["mystery.blob"])["files"][0]
        self.assertTrue(item["needs_review"])
        self.assertIn("Other/Review",item["suggested_path"])

    def test_persian_topic_is_preserved(self):
        item=propose("عکس‌های خانوادگی",["family.jpg"])["files"][0]
        self.assertTrue(item["suggested_path"].startswith("عکس‌های-خانوادگی/"))

    def test_model_gets_metadata_only_and_valid_path_is_accepted(self):
        seen={}
        def proposer(topic,metadata):
            seen["metadata"]=metadata
            return {"files":[{"original_name":"plan.pdf","suggested_path":"Work/Plans/plan.pdf","confidence":0.91,"needs_review":False}]}
        result=propose("Work",["plan.pdf"],proposer)
        self.assertEqual(seen["metadata"],[{"name":"plan.pdf","extension":".pdf"}])
        self.assertTrue(result["model_used"])
        self.assertEqual(result["files"][0]["suggested_path"],"Work/Plans/plan.pdf")

    def test_unsafe_model_path_is_rejected(self):
        def proposer(topic,metadata):
            return {"files":[{"original_name":"plan.pdf","suggested_path":"../../outside/plan.pdf","confidence":0.99,"needs_review":False}]}
        result=propose("Work",["plan.pdf"],proposer)
        self.assertFalse(result["model_used"])
        self.assertNotIn("..",result["files"][0]["suggested_path"])

    def test_duplicate_names_and_path_inputs_are_rejected(self):
        with self.assertRaises(ValueError): propose("Work",["same.txt","same.txt"])
        with self.assertRaises(ValueError): propose("Work",["folder/file.txt"])

    def test_preview_is_non_destructive(self):
        safety=propose("Work",["readme.md"])["safety"]
        self.assertTrue(safety["preview_only"])
        self.assertFalse(safety["file_contents_read"])
        self.assertFalse(safety["files_modified"])

if __name__ == "__main__": unittest.main()
