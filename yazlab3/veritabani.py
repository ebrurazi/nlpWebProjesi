import pymongo
import fasttext
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from pymongo import MongoClient
from passlib.hash import pbkdf2_sha256
from django import forms
from django.contrib.auth.models import User
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django import forms
from django.contrib.auth.models import User
from django.contrib.auth.forms import UserCreationForm
from django import forms
from django.contrib.auth.models import User
from django.contrib.auth.forms import UserCreationForm
import fasttext
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from pymongo import MongoClient
from passlib.hash import pbkdf2_sha256
from .models import Profile

class SignUpForm(UserCreationForm):
    GENDER_CHOICES = [
        ('M', 'Male'),
        ('F', 'Female'),
        ('O', 'Other'),
    ]
    
    age = forms.IntegerField(required=True, label="Yaş")
    location = forms.CharField(max_length=100, required=True, label="Konum")
    gender = forms.ChoiceField(choices=GENDER_CHOICES, required=True, label="Cinsiyet")
    birth_date = forms.DateField(widget=forms.DateInput(attrs={'type': 'date'}), required=True, label="Doğum Tarihi")
    interests = forms.CharField(widget=forms.Textarea, required=True, label="Akademik İlgi Alanları") 

    ft_model = fasttext.load_model('/Users/Desktop/cc.en.300.bin')
    tokenizer = AutoTokenizer.from_pretrained('/Users/Desktop/models/scibert_tokenizer')
    model = AutoModel.from_pretrained('/Users/Desktop/models/scibert_model')

    class Meta:
        model = User
        fields = ('username', 'email', 'password1', 'password2', 'age', 'location', 'interests', 'gender', 'birth_date')

    def save(self, commit=True):
        user = super().save(commit=False)
        if commit:
            user.save()
            profile = Profile.objects.create(
                user=user,
                age=self.cleaned_data['age'],
                location=self.cleaned_data['location'],
                gender=self.cleaned_data['gender'],
                birth_date=self.cleaned_data['birth_date'],
                interests=self.cleaned_data['interests'],
                fasttext_vectors=self.calculate_fasttext_vectors(self.cleaned_data['interests']),
                scibert_vectors=self.calculate_scibert_vectors(self.cleaned_data['interests']),
                fasttext_tp=0,
                fasttext_fp=0,
                scibert_tp=0,
                scibert_fp=0,
                scibert_fn= 0,
                fasttext_fn= 0,
            )
            profile.save()
            self.save_to_mongodb(profile)
        return user

    @staticmethod
    def calculate_fasttext_vectors(interests):
        words = interests.split()
        vectors = [SignUpForm.ft_model.get_word_vector(word) for word in words]
        average_vector = np.mean(vectors, axis=0)
        return average_vector.tolist()

    @staticmethod
    def calculate_scibert_vectors(interests):
        inputs = SignUpForm.tokenizer(interests, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            outputs = SignUpForm.model(**inputs)

        embeddings = outputs.last_hidden_state
        mask = inputs['attention_mask'].unsqueeze(-1).expand(embeddings.size()).float()
        sum_embeddings = torch.sum(embeddings * mask, 1)
        sum_mask = torch.clamp(mask.sum(1), min=1e-9)
        average_embeddings = sum_embeddings / sum_mask

        return average_embeddings.squeeze().tolist()

    def save_to_mongodb(self, profile):
        connection_string = "mongodbclient"
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
            'fasttext_vectors': profile.fasttext_vectors,
            'scibert_vectors': profile.scibert_vectors,
            "fasttext_tp": profile.fasttext_tp,
            "fasttext_fp": profile.fasttext_fp,
            "scibert_tp": profile.scibert_tp,
            "scibert_fp": profile.scibert_fp,
            "scibert_fn": profile.scibert_fn,
            "fasttext_fn": profile.fasttext_fn,
        }

        hashed_password = pbkdf2_sha256.hash(user_data['password'])
        user_data['password'] = hashed_password

        collection.insert_one(user_data)



class LoginForm(AuthenticationForm):
    def __init__(self, request=None, *args, **kwargs):
        super(LoginForm, self).__init__(request=request, *args, **kwargs)
    username = forms.CharField()
    password = forms.CharField(widget=forms.PasswordInput)

    class Meta:
        model = User
        fields = ['username', 'password']