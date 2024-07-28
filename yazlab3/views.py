import os
from bson import ObjectId
from django.contrib.auth import login, logout
from django.shortcuts import render, redirect
from django.db.models import Avg
import numpy as np
from torch import cosine_similarity
from django.http import HttpResponse
from yazlab3.veritabani import SignUpForm
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.contrib import messages
from django.shortcuts import render, get_object_or_404
import os
from django.contrib.auth import login, logout
from django.shortcuts import render, redirect
from django.db.models import Avg
import numpy as np
from torch import cosine_similarity
from django.http import HttpResponse
from yazlab3.veritabani import SignUpForm
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.contrib import messages
import datetime
from pymongo import MongoClient
from .models import Profile
from django.shortcuts import render, redirect
from django.contrib import messages
from pymongo import MongoClient
from .forms import UserForm, ProfileForm


def signup(request):
    if request.method == 'POST':
        form = SignUpForm(request.POST)
        if form.is_valid():
            user = form.save()
            profile = user.profile
            profile.age = form.cleaned_data['age']
            profile.location = form.cleaned_data['location']
            profile.gender = form.cleaned_data['gender']
            profile.birth_date = form.cleaned_data['birth_date']
            profile.interests = form.cleaned_data['interests']
            profile.fasttext_vectors = form.calculate_fasttext_vectors(form.cleaned_data['interests'])
            profile.scibert_vectors = form.calculate_scibert_vectors(form.cleaned_data['interests'])
            profile.save()

            login(request, user)
            return redirect('recommendations')  # Kullanıcıyı başka bir sayfaya yönlendir
    else:
        form = SignUpForm()
    return render(request, 'signup.html', {'form': form})

def home(request):
    return render(request, 'home.html')

@login_required
def update_profile(request):
    if request.method == 'POST':
        user_form = UserForm(request.POST, instance=request.user)
        profile_form = ProfileForm(request.POST, instance=request.user.profile)
        
        if user_form.is_valid() and profile_form.is_valid():
            user_form.save()
            profile = profile_form.save(commit=False)
            profile.user = request.user
            profile.save()
            save_to_mongodb(profile)
            messages.success(request, 'Profiliniz başarıyla güncellendi!')
            return redirect('recommendations')
    else:
        user_form = UserForm(instance=request.user)
        profile_form = ProfileForm(instance=request.user.profile)
    
    return render(request, 'update_profile.html', {
        'user_form': user_form,
        'profile_form': profile_form
    })




def article_detail(request, article_id):
    client = MongoClient('mongodb+srv://zehra:1234@cluster0.n8aotxw.mongodb.net/')
    db = client['yazlab3']
    collection = db['deneme1']
    article = collection.find_one({'_id': ObjectId(article_id)})

    if not article:
        return render(request, 'error.html', {'message': 'Makale bulunamadı.'})

    context = {
        'title': article['title'],
        'summary': article['summary'],
        'keywords': article.get('key_content', 'No keywords'),
    }

    return render(request, 'article_detail.html', context)


def save_to_mongodb(profile):
    connection_string = "mongodb+srv://zehra:1234@cluster0.n8aotxw.mongodb.net/"
    client = MongoClient(connection_string)
    db = client["yazlab3"]
    collection = db["yazlab33"]

    user_data = {
        'username': profile.user.username,
        'email': profile.user.email,
        'password': profile.user.password,
        'age': profile.age,
        'location': profile.location,
        'gender': profile.gender,
        'birth_date': profile.birth_date.isoformat() if profile.birth_date else None,
        'interests': profile.interests,

    }

    # Kullanıcıyı username üzerinden bulup güncelle
    collection.update_one({'username': profile.user.username}, {"$set": user_data}, upsert=True)


def search(request):
    query = request.GET.get('search_query', '')
    results = search(query)
    return render(request, 'search.html', {'results': results})

