from django.db import models

LANGUAGE_CHOICES = [
    ('tamil', 'Tamil'),
    ('telugu', 'Telugu'),
    ('hindi', 'Hindi'),
    ('french', 'French'),
    ('japanese', 'Japanese'),
]

class DetectedObject(models.Model):
    name = models.CharField(max_length=100, unique=True)
    seen = models.BooleanField(default=True)
    learnt = models.BooleanField(default=False)
    languages_learnt_in = models.JSONField(default=list)  # Stores multiple languages

    def __str__(self):
        return self.name

class Score(models.Model):
    points = models.IntegerField(default=0)