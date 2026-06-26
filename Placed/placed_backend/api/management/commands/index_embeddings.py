"""
python manage.py index_embeddings

DB의 모든 Place를 순회하며 Gemini 임베딩을 생성하고
Elasticsearch에 직접 업데이트합니다.

django-elasticsearch-dsl 8.2 에서 prepare_* 메서드가
dense_vector 필드와 연동되지 않는 문제를 우회합니다.
"""

from django.core.management.base import BaseCommand
from elasticsearch import Elasticsearch
from django.conf import settings
from api.models import Place
from api.gemini_service import get_embedding


class Command(BaseCommand):
    help = 'Gemini 임베딩을 생성하여 Elasticsearch에 저장합니다.'

    def handle(self, *args, **options):
        es = Elasticsearch(
            hosts=[settings.ELASTICSEARCH_DSL['default']['hosts']],
            verify_certs=False,
        )

        places = Place.objects.all()
        total = places.count()
        self.stdout.write(f'총 {total}개 장소 임베딩 시작...\n')

        success, failed = 0, 0

        for i, place in enumerate(places, 1):
            try:
                text = f"{place.name} {place.description} {place.address}".strip()
                vector = get_embedding(text)

                es.update(
                    index='places',
                    id=str(place.id),
                    body={'doc': {'embedding': vector}},
                )

                success += 1
                self.stdout.write(f'[{i}/{total}] ✓ {place.name}')

            except Exception as e:
                failed += 1
                self.stdout.write(f'[{i}/{total}] ✗ {place.name} → {e}')

        self.stdout.write(f'\n완료: 성공 {success}개 / 실패 {failed}개')