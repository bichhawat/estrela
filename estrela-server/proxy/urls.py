from django.conf import settings
from django.conf.urls import include, url


urlpatterns = [

    # API (v1)
    url(r'^', include('permissions.accounts.urls')),
    url(r'^', include('permissions.credits.urls')),
    url(r'^', include('permissions.posts.urls')),
    url(r'^', include('permissions.private_messages.urls')),
    url(r'^', include('permissions.replies.urls')),
    url(r'^', include('permissions.user_roles.urls')),
    url(r'^', include('permissions.votes.urls')),

]

if settings.DEBUG:
    import debug_toolbar

    urlpatterns += [
        url(r'^__debug__/', include(debug_toolbar.urls)),
    ]
