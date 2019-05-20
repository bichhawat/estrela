from .norestlib import norest_permission
from .user_roles.models.administrator import Administrator

@norest_permission
def full_permissions(storage, user=None, obj=None, get=None, post=None):
    return True, {}


@norest_permission
def follower_only_permissions(storage, user=None, obj=None, get=None, post=None):
    # User None means AnonymousUser
    if user.is_anonymous:
        return False, {}

    if 'following_ids' in storage:
        following_ids = storage['following_ids']
    else:
        following_ids = [u.id for u in user.following.all()]
        storage['following_ids'] = following_ids
    # print(following_ids)
    user_id = obj['user']['id']
    # print(user_id)

    if user_id in following_ids or user_id == user.id:
        return True, {}
    else:
        return False, {}
        # return False, {'body': '[REDACTED] Follow ' + obj['user']['email'] + ' to see the contents of this post'}

@norest_permission
def test_permission(storage, user=None, obj=None, get=None, post=None):
    if user is None:
        return False, {}

    return False, {'title': "TESTING123"}
