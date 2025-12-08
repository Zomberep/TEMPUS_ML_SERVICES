from transformers import AutoTokenizer, AutoModel
from model import AdvancedRegressionMLP
from threading import Thread

import numpy as np
import torch
import time
import json
import pika

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RUBERT_NAME = "./rubert-tiny2"
tokenizer = AutoTokenizer.from_pretrained(RUBERT_NAME)
rubert_model = AutoModel.from_pretrained(RUBERT_NAME).to(DEVICE)
rubert_model.eval()

NUMBER_OF_CLASSES = 13
INPUT_DIM = 325
HIDDEN_DIM = 256

checkpoint = torch.load("regression_model.pth", map_location=DEVICE)

mlp_model = AdvancedRegressionMLP(
    input_dim=INPUT_DIM,
    hidden_dim=HIDDEN_DIM,
).to(DEVICE)

mlp_model.load_state_dict(checkpoint['model_state_dict'])
mlp_model.eval()

class DurationPredictor:
    ordered_classes = ['Быт', 'Встречи', 'Документы', 'Здоровье', 'Красота', 'Образование', 'Отношения', 'Покупки', 'Путешествия', 'Работа', 'Развлечения', 'Спорт', 'Хобби']

    def predict_duration(self, text: str, classes: list[str]) -> np.int64:
        emb = self.get_embedding(text)

        extra_part = np.zeros(NUMBER_OF_CLASSES)
        for i, cls in enumerate(self.ordered_classes):
            extra_part[i] = 1 if cls in classes else 0

        combined = np.concatenate([emb, extra_part])
        emb_tensor = torch.tensor(combined, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            dur = mlp_model(emb_tensor)

        return np.round(dur.item())


    @staticmethod
    def get_embedding(text: str) -> np.ndarray:
        inputs = tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt"
        ).to(DEVICE)
        with torch.no_grad():
            outputs = rubert_model(**inputs)
        return outputs.last_hidden_state[0, 0].cpu().numpy()

predictor = DurationPredictor()

def process_message(channel, method, properties, body):
    try:
        # 1. Парсинг входного сообщения
        request_data = json.loads(body.decode())
        user_input = request_data["userInput"]
        categories = request_data["categories"]

        # 2. Предсказание
        duration = predictor.predict_duration(user_input, categories)

        # 3. Формирование ответа
        response = {
            "timeInMinutes": duration,
        }

        # 4. Отправка ответа обратно
        channel.basic_publish(
            exchange="tempus",
            routing_key=properties.reply_to,  # очередь отправителя
            properties=pika.BasicProperties(
                correlation_id=properties.correlation_id
            ),
            body=json.dumps(response, ensure_ascii=False)
        )

        # Подтверждение обработки сообщения
        channel.basic_ack(delivery_tag=method.delivery_tag)

    except Exception as e:
        print(f"Ошибка обработки: {e}")
        channel.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

def start_rabbitmq_listener():
    try:
        time.sleep(5)

        # Параметры подключения (измените под ваш RabbitMQ)
        connection = pika.BlockingConnection(
            pika.ConnectionParameters(host="rmq", port=5672)  # или "rabbitmq" в Docker
        )
        channel = connection.channel()

        # Объявление exchange (если не существует)
        channel.exchange_declare(exchange="tempus", exchange_type="topic", durable=True)

        result = channel.queue_declare(queue="ml-duration")
        queue_name = result.method.queue

        # Привязка к нужному topic
        channel.queue_bind(
            exchange="tempus",
            queue=queue_name,
            routing_key='ml.calculate-time.command'
        )

        # Подписка
        channel.basic_consume(
            queue=queue_name,
            on_message_callback=process_message,
            auto_ack=False
        )

        channel.start_consuming()

    except Exception as e:
        print(f"❌ RabbitMQ thread crashed: {e}")
        raise

if __name__ == "__main__":
    rabbit_thread = Thread(target=start_rabbitmq_listener, daemon=True)
    rabbit_thread.start()

    print("🚀 ML Category Service is running. Press Ctrl+C to exit.")

    try:
        while True:
            time.sleep(1)  # чтобы не нагружать CPU
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")