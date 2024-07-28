from django.shortcuts import render
from pymongo import MongoClient
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from operator import itemgetter
from bson.objectid import ObjectId
from django.http import JsonResponse
from django.contrib.auth.decorators import login_required
import logging
logger = logging.getLogger(__name__)

def is_valid_vector(vector):
    if vector is None:
        return False
    return not np.any(np.isnan(vector))

@login_required
def recommendations(request):
    client = MongoClient('mongodbclient//')
    db = client['yazlab3']
    collection_users = db['yazlab33']
    collection_texts = db['deneme1']

    user_data = collection_users.find_one({'username': request.user.username})
    if user_data is None:
        return render(request, 'error.html', {'message': 'Kullanıcı verisi eksik'})

    texts = list(collection_texts.find({}))

    similarities_fs = []
    similarities_sc = []
    for text in texts:
        if 'fasttext_vector' in text and 'scibert_vector' in text and is_valid_vector(text['fasttext_vector']) and is_valid_vector(text['scibert_vector']):
            text_vector_fs = np.array(text['fasttext_vector']).reshape(1, -1)
            text_vector_sc = np.array(text['scibert_vector']).reshape(1, -1)
            similarity_fs = cosine_similarity(np.array(user_data['fasttext_vectors']).reshape(1, -1), text_vector_fs)[0][0]
            similarity_sc = cosine_similarity(np.array(user_data['scibert_vectors']).reshape(1, -1), text_vector_sc)[0][0]
            similarities_fs.append((similarity_fs, text['title'], text['summary'], text.get('key_content', 'No keywords'), str(text['_id'])))
            similarities_sc.append((similarity_sc, text['title'], text['summary'], text.get('key_content', 'No keywords'), str(text['_id'])))

    top_texts_fs = sorted(similarities_fs, key=itemgetter(0), reverse=True)[:6]
    top_texts_sc = sorted(similarities_sc, key=itemgetter(0), reverse=True)[:6]

    collection_users.update_one({'_id': user_data['_id']}, {'$set': {
        'current_recommendations': {
            'fasttext': [article_id for _, _, _, _, article_id in top_texts_fs],
            'scibert': [article_id for _, _, _, _, article_id in top_texts_sc]
        }
    }})

    # Precision ve recall değerlerini hesapla
    fasttext_tp = user_data.get('fasttext_tp', 0)
    fasttext_fp = user_data.get('fasttext_fp', 0)
    scibert_tp = user_data.get('scibert_tp', 0)
    scibert_fp = user_data.get('scibert_fp', 0)
    scibert_fn = user_data.get('scibert_fn', 0)
    fasttext_fn = user_data.get('fasttext_fn', 0)

    fasttext_precision = fasttext_tp / (fasttext_tp + fasttext_fp) if (fasttext_tp + fasttext_fp) > 0 else 0
    fasttext_recall = fasttext_tp / (fasttext_tp + fasttext_fn) if (fasttext_tp + fasttext_fn) > 0 else 0
    scibert_precision = scibert_tp / (scibert_tp + scibert_fp) if (scibert_tp + scibert_fp) > 0 else 0
    scibert_recall = scibert_tp / (scibert_tp + scibert_fn) if (scibert_tp + scibert_fn) > 0 else 0

    context = {
        'username': request.user.username,
        'fasttext_texts': top_texts_fs[:5],
        'scibert_texts': top_texts_sc[:5],
        'fasttext_extra': top_texts_fs[5] if len(top_texts_fs) > 5 else None,
        'scibert_extra': top_texts_sc[5] if len(top_texts_sc) > 5 else None,
        'fasttext_precision': fasttext_precision,
        'fasttext_recall': fasttext_recall,
        'scibert_precision': scibert_precision,
        'scibert_recall': scibert_recall
    }

    return render(request, 'recommendations.html', context)
@login_required
def update_interest_and_recommendations(request, article_id):
    if not request.user.is_authenticated:
        return JsonResponse({'status': 'error', 'message': 'Authentication required'}, status=401)

    client = MongoClient('mongodbclient///')
    db = client['yazlab3']
    collection_users = db['yazlab33']
    collection_texts = db['deneme1']

    user_data = collection_users.find_one({'username': request.user.username})
    article_data = collection_texts.find_one({'_id': ObjectId(article_id)})

    if not user_data or not article_data:
        return JsonResponse({'status': 'error', 'message': 'User or article not found'}, status=404)

    feedback_type = request.POST.get('feedback_type')
    if feedback_type not in ['positive', 'negative']:
        return JsonResponse({'status': 'error', 'message': 'Invalid feedback type'}, status=400)

    model_type = request.POST.get('model_type')
    if model_type not in ['fasttext', 'scibert']:
        return JsonResponse({'status': 'error', 'message': 'Invalid model type'}, status=400)

    # Kullanıcı vektörlerini güncelle
    user_vector_fs = np.array(user_data['fasttext_vectors'])
    article_vector_fs = np.array(article_data['fasttext_vector'])
    new_vector_fs = (user_vector_fs + article_vector_fs) / 2 if is_valid_vector(user_vector_fs) and is_valid_vector(article_vector_fs) else user_vector_fs

    user_vector_sc = np.array(user_data['scibert_vectors'])
    article_vector_sc = np.array(article_data['scibert_vector'])
    new_vector_sc = (user_vector_sc + article_vector_sc) / 2 if is_valid_vector(user_vector_sc) and is_valid_vector(article_vector_sc) else user_vector_sc

    collection_users.update_one({'_id': user_data['_id']}, {'$set': {
        'fasttext_vectors': new_vector_fs.tolist(),
        'scibert_vectors': new_vector_sc.tolist()
    }})

    # Geri bildirim işlemleri
    if feedback_type == 'positive':
        if model_type == 'fasttext':
            collection_users.update_one({'_id': user_data['_id']}, {'$inc': {'fasttext_tp': 1}})
        elif model_type == 'scibert':
            collection_users.update_one({'_id': user_data['_id']}, {'$inc': {'scibert_tp': 1}})
    elif feedback_type == 'negative':
        if model_type == 'fasttext':
            collection_users.update_one({'_id': user_data['_id']}, {'$inc': {'fasttext_fn': 1}})
        elif model_type == 'scibert':
            collection_users.update_one({'_id': user_data['_id']}, {'$inc': {'scibert_fn': 1}})
        # FN (False Negative) değerini artır
        collection_users.update_one({'_id': user_data['_id']}, {'$inc': {f'{model_type}_fn': 1}})

    # Yeni önerileri güncelle
    texts = list(collection_texts.find({}))
    similarities_fs = []
    similarities_sc = []

    for text in texts:
        if 'fasttext_vector' in text and 'scibert_vector' in text and is_valid_vector(text['fasttext_vector']) and is_valid_vector(text['scibert_vector']):
            text_vector_fs = np.array(text['fasttext_vector']).reshape(1, -1)
            text_vector_sc = np.array(text['scibert_vector']).reshape(1, -1)
            similarity_fs = cosine_similarity(new_vector_fs.reshape(1, -1), text_vector_fs)[0][0]
            similarity_sc = cosine_similarity(new_vector_sc.reshape(1, -1), text_vector_sc)[0][0]
            similarities_fs.append((similarity_fs, text['title'], text['summary'], text.get('key_content', 'No keywords'), str(text['_id'])))
            similarities_sc.append((similarity_sc, text['title'], text['summary'], text.get('key_content', 'No keywords'), str(text['_id'])))

    top_texts_fs = sorted(similarities_fs, key=itemgetter(0), reverse=True)[:5]
    top_texts_sc = sorted(similarities_sc, key=itemgetter(0), reverse=True)[:5]

    # Precision ve recall hesaplamaları
    fasttext_tp = user_data.get('fasttext_tp', 0)
    fasttext_fp = user_data.get('fasttext_fp', 0)
    scibert_tp = user_data.get('scibert_tp', 0)
    scibert_fp = user_data.get('scibert_fp', 0)
    scibert_fn = user_data.get('scibert_fn', 0)
    fasttext_fn = user_data.get('fasttext_fn', 0)

    fasttext_precision = fasttext_tp / (fasttext_tp + fasttext_fp) if (fasttext_tp + fasttext_fp) > 0 else 0
    fasttext_recall = fasttext_tp / (fasttext_tp + fasttext_fn) if (fasttext_tp + fasttext_fn) > 0 else 0
    scibert_precision = scibert_tp / (scibert_tp + scibert_fp) if (scibert_tp + scibert_fp) > 0 else 0
    scibert_recall = scibert_tp / (scibert_tp + scibert_fn) if (scibert_tp + scibert_fn) > 0 else 0

    return JsonResponse({
        'status': 'success',
        'message': 'Recommendations updated!',
        'fasttext_texts': top_texts_fs,
        'scibert_texts': top_texts_sc,
        'fasttext_precision': fasttext_precision,
        'fasttext_recall': fasttext_recall,
        'scibert_precision': scibert_precision,
        'scibert_recall': scibert_recall
    })

@login_required
def replace_article(request, article_id, type):
    if not request.user.is_authenticated:
        return JsonResponse({'status': 'error', 'message': 'Authentication required'}, status=401)

    client = MongoClient('mongodbclient')
    db = client['yazlab3']
    collection_users = db['yazlab33']
    collection_texts = db['deneme1']

    user_data = collection_users.find_one({'username': request.user.username})

    if not user_data or 'current_recommendations' not in user_data:
        return JsonResponse({'status': 'error', 'message': 'User or recommendations not found'}, status=404)

    recommendations = user_data['current_recommendations'].get(type, [])

    if article_id in recommendations:
        recommendations.remove(article_id)

    all_texts = list(collection_texts.find({}))

    similarities = []
    for text in all_texts:
        if 'fasttext_vector' in text and 'scibert_vector' in text and is_valid_vector(text['fasttext_vector']) and is_valid_vector(text['scibert_vector']):
            text_vector = np.array(text[f'{type}_vector']).reshape(1, -1)
            user_vector = np.array(user_data[f'{type}_vectors']).reshape(1, -1)
            similarity = cosine_similarity(user_vector, text_vector)[0][0]
            if str(text['_id']) not in recommendations:
                similarities.append((similarity, text['title'], text['summary'], text.get('key_content', 'No keywords'), str(text['_id'])))

    similarities = sorted(similarities, key=itemgetter(0), reverse=True)

    if similarities:
        new_article = similarities[0]
        recommendations.append(new_article[4])
        recommendations = recommendations[:5]
        collection_users.update_one({'_id': user_data['_id']}, {'$set': {f'current_recommendations.{type}': recommendations}})

        if type == 'fasttext':
            collection_users.update_one({'_id': user_data['_id']}, {'$inc': {'fasttext_fp': 1}})
        elif type == 'scibert':
            collection_users.update_one({'_id': user_data['_id']}, {'$inc': {'scibert_fp': 1}})

        # FN (False Negative) değerini artır
        collection_users.update_one({'_id': user_data['_id']}, {'$inc': {f'{type}_fn': 1}})

        fasttext_tp = user_data.get('fasttext_tp', 0)
        fasttext_fp = user_data.get('fasttext_fp', 0)
        scibert_tp = user_data.get('scibert_tp', 0)
        scibert_fp = user_data.get('scibert_fp', 0)
        scibert_fn = user_data.get('scibert_fn', 0)
        fasttext_fn = user_data.get('fasttext_fn', 0)

        fasttext_precision = fasttext_tp / (fasttext_tp + fasttext_fp) if (fasttext_tp + fasttext_fp) > 0 else 0
        fasttext_recall = fasttext_tp / (fasttext_tp + fasttext_fn) if (fasttext_tp + fasttext_fn) > 0 else 0
        scibert_precision = scibert_tp / (scibert_tp + scibert_fp) if (scibert_tp + scibert_fp) > 0 else 0
        scibert_recall = scibert_tp / (scibert_tp + scibert_fn) if (scibert_tp + scibert_fn) > 0 else 0

        return JsonResponse({
            'status': 'success',
            'message': 'Article removed from interests!',
            'article': new_article,
            'fasttext_precision': fasttext_precision,
            'fasttext_recall': fasttext_recall,
            'scibert_precision': scibert_precision,
            'scibert_recall': scibert_recall
        })
    else:
        return JsonResponse({'status': 'error', 'message': 'No new articles available'}, status=404)


@login_required
def article_detail(request, article_id):
    client = MongoClient('mongodbclient//')
    db = client['yazlab3']
    collection_users = db['yazlab33']
    collection_texts = db['deneme1']

    user_data = collection_users.find_one({'username': request.user.username})
    article_data = collection_texts.find_one({'_id': ObjectId(article_id)})

    if not user_data or not article_data:
        return JsonResponse({'status': 'error', 'message': 'User or article not found'}, status=404)

    # Kullanıcı vektörlerini güncelle
    user_vector_fs = np.array(user_data['fasttext_vectors'])
    article_vector_fs = np.array(article_data['fasttext_vector'])
    new_vector_fs = (user_vector_fs + article_vector_fs) / 2 if is_valid_vector(user_vector_fs) and is_valid_vector(article_vector_fs) else user_vector_fs

    user_vector_sc = np.array(user_data['scibert_vectors'])
    article_vector_sc = np.array(article_data['scibert_vector'])
    new_vector_sc = (user_vector_sc + article_vector_sc) / 2 if is_valid_vector(user_vector_sc) and is_valid_vector(article_vector_sc) else user_vector_sc

    collection_users.update_one({'_id': user_data['_id']}, {'$set': {
        'fasttext_vectors': new_vector_fs.tolist(),
        'scibert_vectors': new_vector_sc.tolist()
    }})

    # Yeni önerileri güncelle
    texts = list(collection_texts.find({}))
    similarities_fs = []
    similarities_sc = []

    for text in texts:
        if 'fasttext_vector' in text and 'scibert_vector' in text and is_valid_vector(text['fasttext_vector']) and is_valid_vector(text['scibert_vector']):
            text_vector_fs = np.array(text['fasttext_vector']).reshape(1, -1)
            text_vector_sc = np.array(text['scibert_vector']).reshape(1, -1)
            similarity_fs = cosine_similarity(new_vector_fs.reshape(1, -1), text_vector_fs)[0][0]
            similarity_sc = cosine_similarity(new_vector_sc.reshape(1, -1), text_vector_sc)[0][0]
            similarities_fs.append((similarity_fs, text['title'], text['summary'], text.get('key_content', 'No keywords'), str(text['_id'])))
            similarities_sc.append((similarity_sc, text['title'], text['summary'], text.get('key_content', 'No keywords'), str(text['_id'])))

    top_texts_fs = sorted(similarities_fs, key=itemgetter(0), reverse=True)[:5]
    top_texts_sc = sorted(similarities_sc, key=itemgetter(0), reverse=True)[:5]

    # Precision ve recall hesaplamaları
    fasttext_tp = user_data.get('fasttext_tp', 0)
    fasttext_fp = user_data.get('fasttext_fp', 0)
    scibert_tp = user_data.get('scibert_tp', 0)
    scibert_fp = user_data.get('scibert_fp', 0)
    scibert_fn = user_data.get('scibert_fn', 0)
    fasttext_fn = user_data.get('fasttext_fn', 0)

    fasttext_precision = fasttext_tp / (fasttext_tp + fasttext_fp) if (fasttext_tp + fasttext_fp) > 0 else 0
    fasttext_recall = fasttext_tp / (fasttext_tp + fasttext_fn) if (fasttext_tp + fasttext_fn) > 0 else 0
    scibert_precision = scibert_tp / (scibert_tp + scibert_fp) if (scibert_tp + scibert_fp) > 0 else 0
    scibert_recall = scibert_tp / (scibert_tp + scibert_fn) if (scibert_tp + scibert_fn) > 0 else 0

    return JsonResponse({
        'status': 'success',
        'message': 'Recommendations updated!',
        'fasttext_texts': top_texts_fs,
        'scibert_texts': top_texts_sc,
        'fasttext_precision': fasttext_precision,
        'fasttext_recall': fasttext_recall,
        'scibert_precision': scibert_precision,
        'scibert_recall': scibert_recall
    })
@login_required
def update_user_vector(request, article_id):
    client = MongoClient('mongodbclientt//')
    db = client['yazlab3']
    collection_users = db['yazlab33']
    collection_texts = db['deneme1']

    user_data = collection_users.find_one({'username': request.user.username})
    article_data = collection_texts.find_one({'_id': ObjectId(article_id)})

    if not user_data or not article_data:
        return JsonResponse({'status': 'error', 'message': 'User or article not found'}, status=404)

    model_type = request.POST.get('model_type')
    if model_type not in ['fasttext', 'scibert']:
        return JsonResponse({'status': 'error', 'message': 'Invalid model type'}, status=400)

    # Kullanıcı vektörlerini güncelle
    user_vector_fs = np.array(user_data['fasttext_vectors'])
    article_vector_fs = np.array(article_data['fasttext_vector'])
    new_vector_fs = (user_vector_fs + article_vector_fs) / 2 if is_valid_vector(user_vector_fs) and is_valid_vector(article_vector_fs) else user_vector_fs

    user_vector_sc = np.array(user_data['scibert_vectors'])
    article_vector_sc = np.array(article_data['scibert_vector'])
    new_vector_sc = (user_vector_sc + article_vector_sc) / 2 if is_valid_vector(user_vector_sc) and is_valid_vector(article_vector_sc) else user_vector_sc

    if model_type == 'fasttext':
        collection_users.update_one({'_id': user_data['_id']}, {'$inc': {'fasttext_tp': 1}})
    elif model_type == 'scibert':
        collection_users.update_one({'_id': user_data['_id']}, {'$inc': {'scibert_tp': 1}})

    collection_users.update_one({'_id': user_data['_id']}, {'$set': {
        'fasttext_vectors': new_vector_fs.tolist(),
        'scibert_vectors': new_vector_sc.tolist()
    }})

    # Yeni önerileri hesapla
    texts = list(collection_texts.find({}))
    similarities_fs = []
    similarities_sc = []

    for text in texts:
        if 'fasttext_vector' in text and 'scibert_vector' in text and is_valid_vector(text['fasttext_vector']) and is_valid_vector(text['scibert_vector']):
            text_vector_fs = np.array(text['fasttext_vector']).reshape(1, -1)
            text_vector_sc = np.array(text['scibert_vector']).reshape(1, -1)
            similarity_fs = cosine_similarity(new_vector_fs.reshape(1, -1), text_vector_fs)[0][0]
            similarity_sc = cosine_similarity(new_vector_sc.reshape(1, -1), text_vector_sc)[0][0]
            similarities_fs.append((similarity_fs, text['title'], text['summary'], text.get('key_content', 'No keywords'), str(text['_id'])))
            similarities_sc.append((similarity_sc, text['title'], text['summary'], text.get('key_content', 'No keywords'), str(text['_id'])))

    top_texts_fs = sorted(similarities_fs, key=itemgetter(0), reverse=True)[:5]
    top_texts_sc = sorted(similarities_sc, key=itemgetter(0), reverse=True)[:5]

    # Precision ve recall hesaplamaları
    fasttext_tp = user_data.get('fasttext_tp', 0)
    fasttext_fp = user_data.get('fasttext_fp', 0)
    scibert_tp = user_data.get('scibert_tp', 0)
    scibert_fp = user_data.get('scibert_fp', 0)
    scibert_fn = user_data.get('scibert_fn', 0)
    fasttext_fn = user_data.get('fasttext_fn', 0)

    fasttext_precision = fasttext_tp / (fasttext_tp + fasttext_fp) if (fasttext_tp + fasttext_fp) > 0 else 0
    fasttext_recall = fasttext_tp / (fasttext_tp + fasttext_fn) if (fasttext_tp + fasttext_fn) > 0 else 0
    scibert_precision = scibert_tp / (scibert_tp + scibert_fp) if (scibert_tp + scibert_fp) > 0 else 0
    scibert_recall = scibert_tp / (scibert_tp + scibert_fn) if (scibert_tp + scibert_fn) > 0 else 0

    return JsonResponse({
        'status': 'success',
        'message': 'User vector updated and recommendations refreshed!',
        'fasttext_texts': top_texts_fs,
        'scibert_texts': top_texts_sc,
        'fasttext_precision': fasttext_precision,
        'fasttext_recall': fasttext_recall,
        'scibert_precision': scibert_precision,
        'scibert_recall': scibert_recall
    })

def search(request):
    query = request.GET.get('search_query', '')  # Arama formundan gelen sorgu
    results = []
    if query:  # Eğer sorgu boş değilse
        # MongoDB bağlantısını yapılandır
        client = MongoClient('mongodbclient//')
        db = client['yazlab3']
        collection_texts = db['deneme1']

        # Sorguya göre filtrele
        cursor = collection_texts.find({'key_content': {'$regex': query, '$options': 'i'}})
        
        # Cursor'dan gelen belgeleri işle
        results = [{'title': doc['title'], 'summary': doc['summary'], 'key_content': doc['key_content']} for doc in cursor if 'title' in doc and 'summary' in doc and 'key_content']

    # Arama sonuçlarını template'e gönder
    return render(request, 'search.html', {'results': results})
