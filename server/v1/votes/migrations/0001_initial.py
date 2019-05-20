# -*- coding: utf-8 -*-
# Generated by Django 1.11.2 on 2018-09-12 21:43
from __future__ import unicode_literals

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        ('posts', '0001_initial'),
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name='PostVote',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('value', models.IntegerField(choices=[(1, 'Up'), (-1, 'Down')])),
                ('post', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='post_votes', to='posts.Post')),
                ('user', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='post_votes', to=settings.AUTH_USER_MODEL)),
            ],
            options={
                'default_related_name': 'post_votes',
            },
        ),
        migrations.AlterUniqueTogether(
            name='postvote',
            unique_together=set([('post', 'user')]),
        ),
    ]