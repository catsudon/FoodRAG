from deep_translator import GoogleTranslator

from lingua import LanguageDetectorBuilder


class Translator:
    def __init__(self, target="en"):
        self.target = target
        self.__translator_en = GoogleTranslator(source="auto", target=target)

    def translate(self, text: str):
        return self.__translator_en.translate(text)


if __name__ == "__main__":
    translator = Translator()
    print(translator.translate("Hello, how are you", "zh"))
