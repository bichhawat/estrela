from django.conf.urls import url
from permissions.views import post_view
from permissions.permissions import full_permissions


urlpatterns = [

    # Posts
    url(r'^posts$', post_view),
    url(r'^posts/(?P<post_id>[\d]+)$', full_permissions),

]
