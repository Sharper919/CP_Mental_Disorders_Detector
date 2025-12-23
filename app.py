import streamlit as st
import torch
import re
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification


class TextPreprocessor:
    @staticmethod
    def clean(text: str) -> str:
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#(\w+)', r'\1', text)
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'[^a-z\s]', '', text)
        return text


class TransformerModel:
    def __init__(self, model_path: str = "models"):
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_path)
        self.model = DistilBertForSequenceClassification.from_pretrained(model_path)
        self.model.eval()

    def predict(self, text: str):
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True
        )

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred_class].item()

        return pred_class, confidence


class MentalDisordersApp:
    def __init__(self):
        st.set_page_config(page_title="Mental Health Detector")
        st.title("Виявлення психоемоційних порушень за текстом")

        self.preprocessor = TextPreprocessor()
        self.model = self.load_model()

        self.tab_analyze, self.tab_info = st.tabs(
            ["Аналіз тексту", "Про систему"]
        )

    @staticmethod
    @st.cache_resource
    def load_model():
        return TransformerModel()


    def run(self):
        self.render_analyze_tab()
        self.render_info_tab()

    def render_analyze_tab(self):
        with self.tab_analyze:
            user_input = st.text_area("Введіть текст для аналізу:", height=150)

            if st.button("Аналізувати"):
                if not user_input.strip():
                    st.warning("Введіть текст")
                    return

                cleaned_text = self.preprocessor.clean(user_input)
                pred_class, confidence = self.model.predict(cleaned_text)

                if pred_class == 1:
                    st.error(
                        f"Ознаки психоемоційних проблем "
                        f"(ймовірність {confidence:.2%})"
                    )
                else:
                    st.success(
                        f"Ознак психоемоційних проблем не виявлено "
                        f"(ймовірність {confidence:.2%})"
                    )

    def render_info_tab(self):
        with self.tab_info:
            st.subheader("Психоемоційні порушення")

            st.markdown("""
            **Психоемоційні порушення** — це синдроми, що спричиняють значні порушення поведінки, 
            емоцій та когнітивних функцій. До них належать депресія, параноя, шизофренія, 
            тривожний, біполярний, посттравматичний стресовий розлад тощо.""")

            st.image(
                "images/pr1.jpg",
                use_container_width=True
            )

            st.image(
                "images/pr2.jpg",
                use_container_width=True
            )

            st.markdown("""
            Раннє виявлення таких порушень є надзвичайно важливим, оскільки дозволяє своєчасно 
            надати психологічну або медичну допомогу та знизити ризик ускладнень. В умовах активного 
            використання соціальних мереж та онлайн-комунікацій аналіз текстових повідомлень може 
            використовуватися як ефективний засіб первинної оцінки психоемоційного стану людини.
            """)

            st.markdown("""
            ### Про систему

            Дана система призначена для автоматизованого аналізу текстових повідомлень з метою 
            виявлення можливих ознак психоемоційних порушень. Система використовує методи обробки 
            природної мови та нейромережеві технології для виконання попередньої оцінки психічного 
            стану людини на основі введеного тексту. Основною метою є раннє виявлення 
            психоемоційних порушень у людей для своєчасного надання їм необхідної допомоги.
            """)


            st.markdown("""
            ### Використана модель

            У даній системі використовується **DistilBERT** — компактна трансформерна модель 
            обробки природної мови, яка є оптимізованою версією BERT. Вона здатна ефективно аналізувати 
            текст та враховувати контекст слів у реченнях.
            """)

            st.image(
                "images/dbert.webp",
                use_container_width=True
            )

            st.markdown("""
            **Основні характеристики моделі:**
            - виконує **бінарну класифікацію текстів** (наявність або відсутність психоемоційних проблем);
            - використовує токенізацію з урахуванням контексту;
            - працює швидко та потребує менше ресурсів у порівнянні з класичною моделлю BERT.

            Модель не замінює професійну медичну діагностику, проте може бути використана як 
            допоміжний інструмент для попереднього аналізу текстових даних.
            """)


if __name__ == "__main__":
    app = MentalDisordersApp()
    app.run()
