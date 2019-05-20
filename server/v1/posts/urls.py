from django.conf.urls import include, url
from .views.post import PostView, PostDetail
from django.conf import settings

urlpatterns = [

    # Posts
    url(r'^posts$', PostView.as_view()),
    url(r'^posts/(?P<post_id>[\d]+)$', PostDetail.as_view()),

]
