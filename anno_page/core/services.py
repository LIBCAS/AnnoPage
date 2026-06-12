from uuid import uuid4
from datetime import datetime

class UuidService:
    def __call__(self):
        return self.generate_uuid()

    @staticmethod
    def generate_uuid():
        return str(uuid4())


class DateTimeService:
    def __call__(self, **kwargs):
        return self.get_date_time(**kwargs)

    @staticmethod
    def get_date_time(**kwargs):
        return datetime.now().isoformat(**kwargs)
