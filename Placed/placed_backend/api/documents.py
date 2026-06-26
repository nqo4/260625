from django_elasticsearch_dsl import Document, fields
from django_elasticsearch_dsl.registries import registry
from .models import Place

@registry.register_document
class PlaceDocument(Document):
    class Index:
        name = 'places'
        settings = {
            'number_of_shards': 1,
            'number_of_replicas': 0,
        }
        mappings = {
            'properties': {
                'embedding': {
                    'type': 'dense_vector',
                    'dims': 3072,
                    'index': True,
                    'similarity': 'cosine',
                }
            }
        }

    name = fields.TextField()
    description = fields.TextField()
    address = fields.TextField()

    class Django:
        model = Place
        fields = [
            'id',
            'created_at',
            'image_url',
        ]

    def prepare_embedding(self, instance):
        """
        장소가 ES에 색인될 때 자동으로 호출됩니다.
        name + description + address 를 합쳐 임베딩 벡터를 생성합니다.
        """
        from .gemini_service import get_embedding
        text = f"{instance.name} {instance.description} {instance.address}".strip()
        if not text:
            return [0.0] * 3072 
        return get_embedding(text)