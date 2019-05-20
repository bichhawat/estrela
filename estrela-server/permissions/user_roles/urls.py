from django.conf.urls import url
from permissions.permissions import full_permissions


urlpatterns = [

    # Administrators
    url(r'^administrators$', full_permissions),

    # Moderators
    url(r'^moderators$', full_permissions),
    url(r'^moderators/(?P<moderator_id>[\d]+)$', full_permissions),

]
