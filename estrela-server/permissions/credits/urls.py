from django.conf.urls import url
from permissions.permissions import full_permissions


urlpatterns = [

    # Invitations
    url(r'^invitations$', full_permissions),

    # Transfers
    url(r'^transfers$', full_permissions),

    # Wallets
    url(r'^wallets/(?P<user_id>[\d]+)$', full_permissions),

]
