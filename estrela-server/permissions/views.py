from .permissions import follower_only_permissions
from .norestlib import norest_view

@norest_view
def post_view(request):
    return [follower_only_permissions]
