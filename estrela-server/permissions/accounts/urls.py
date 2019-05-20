from django.conf.urls import url
from permissions.permissions import full_permissions

urlpatterns = [

    # Accept invitation
    url(r'^accept_invitation$', full_permissions),

    # Login / logout
    url(r'^login$', full_permissions),
    url(r'^logout$', full_permissions),

    # Password management
    url(r'^reset_password$', full_permissions),
    url(r'^update_password$', full_permissions),

    # Profiles
    url(r'^profiles$', full_permissions),
    url(r'^profiles/(?P<profile_id>[\d]+)$', full_permissions),

    # Users
    url(r'^users$', full_permissions),
    url(r'^users/(?P<user_id>[\d]+)$', full_permissions),

    # Follow User
    url(r'^follow/(?P<user_id>[\d]+)$', full_permissions),

]
