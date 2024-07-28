from django.db import models
from django.contrib.auth.models import User

class Profile(models.Model):
    GENDER_CHOICES = [
        ('M', 'Male'),
        ('F', 'Female'),
        ('O', 'Other'),
    ]

    user = models.OneToOneField(User, on_delete=models.CASCADE)
    age = models.IntegerField(null=True, blank=True)
    location = models.CharField(max_length=100)
    gender = models.CharField(max_length=1, choices=GENDER_CHOICES)
    birth_date = models.DateField()
    interests = models.TextField()
    fasttext_vectors = models.JSONField()
    scibert_vectors = models.JSONField()
    fasttext_tp = models.IntegerField(default=0)
    fasttext_fp = models.IntegerField(default=0)
    scibert_tp = models.IntegerField(default=0)
    scibert_fp = models.IntegerField(default=0)
    scibert_fn = models.IntegerField(default=0)
    fasttext_fn = models.IntegerField(default=0)

    def __str__(self):
        return self.user.username
