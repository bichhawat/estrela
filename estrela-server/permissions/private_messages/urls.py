from django.conf.urls import url
from permissions.permissions import full_permissions


urlpatterns = [

    # Private messages
    url(r'^private_messages$', full_permissions),
    url(r'^private_messages/(?P<private_message_id>[\d]+)$',
        full_permissions),

]
