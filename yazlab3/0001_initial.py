from django.db import migrations, models
import django.contrib.postgres.fields.jsonb
import django.db.models.deletion

class Migration(migrations.Migration):

    initial = True

    dependencies = [
        ('auth', '0012_alter_user_first_name_max_length'),
    ]

    operations = [
        migrations.CreateModel(
            name='Profile',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('age', models.IntegerField(blank=True, null=True)),
                ('location', models.CharField(max_length=100)),
                ('gender', models.CharField(choices=[('M', 'Male'), ('F', 'Female'), ('O', 'Other')], max_length=1)),
                ('birth_date', models.DateField()),
                ('interests', models.TextField()),
                ('fasttext_vectors', django.contrib.postgres.fields.jsonb.JSONField()),
                ('scibert_vectors', django.contrib.postgres.fields.jsonb.JSONField()),
                ('fasttext_tp', models.IntegerField(default=0)),
                ('fasttext_fp', models.IntegerField(default=0)),
                ('scibert_tp', models.IntegerField(default=0)),
                ('scibert_fp', models.IntegerField(default=0)),
                ('scibert_fn', models.IntegerField(default=0)),
                ('fasttext_fn', models.IntegerField(default=0)),
                ('user', models.OneToOneField(on_delete=django.db.models.deletion.CASCADE, to='auth.user')),
            ],
        ),
    ]
