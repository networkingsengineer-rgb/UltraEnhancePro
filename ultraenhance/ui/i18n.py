"""Very small translation table. Missing keys fall back to English."""

LANGUAGES = ("English", "हिन्दी (Hindi)")

_HI = {
    "Home": "होम", "Photo Enhance": "फ़ोटो सुधार", "Video Enhance": "वीडियो सुधार",
    "Batch Queue": "बैच कतार", "Settings": "सेटिंग्स", "Help": "मदद",
    "Add Files": "फ़ाइलें जोड़ें", "Start": "शुरू करें", "Pause": "रोकें", "Resume": "फिर शुरू",
    "Stop": "बंद करें", "Open Output Folder": "आउटपुट फ़ोल्डर खोलें",
    "File": "फ़ाइल", "Edit": "संपादन", "Process": "प्रोसेस", "Tools": "टूल्स", "View": "दृश्य",
    "Drop photos or videos here": "फ़ोटो या वीडियो यहाँ छोड़ें",
    "Preview": "पूर्वावलोकन", "Add to Queue": "कतार में जोड़ें",
    "Apply settings to selected": "चुने हुए पर सेटिंग्स लागू करें",
    "Apply settings to all": "सभी पर सेटिंग्स लागू करें",
    "Preview 5 seconds": "5 सेकंड पूर्वावलोकन", "Estimate": "अनुमान",
    "Save settings": "सेटिंग्स सहेजें",
}


class Translator:
    def __init__(self, language: str = "English"):
        self.table = _HI if language.startswith("हिन्दी") else {}

    def __call__(self, text: str) -> str:
        return self.table.get(text, text)
