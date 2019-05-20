from django.http import HttpResponse
from django.core import serializers
from functools import wraps
import json


def norest_permission(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        body = args[0]
        # body = json.loads(request.body)
        user = body['norest_user']
        user = user if user else None
        data = body['norest_response']
        get = body['norest_get']
        # print(get)
        get = get if get else None
        post = body['norest_post']
        post = post if post else None

        filtered_data = []
        storage = {}

        for obj in data:
            perm, default = func(storage, user, obj, get, post)
            # import pdb; pdb.set_trace()
            if perm:
                filtered_data.append(obj)
            elif default:
                for key in default:
                    obj[key] = default[key]
                filtered_data.append(obj)

        # print(filtered_data)
        return filtered_data
    return wrapper


def norest_view(*permission_list):
    def outwrap(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            request = args[0] # This is the request sent by the backend
            response = func(*args, **kwargs)
            response._is_rendered = True
            data = response.data
            print('hello')
            response._is_rendered = False

            body = {
                'norest_user': request.user,
                'norest_response': data,
                'norest_get': request.GET.__dict__,
                'norest_post': request.POST.__dict__
            }
            filtered_data = []
            for permission in permission_list:
                filtered_data = permission(body)
                body['norest_response'] = filtered_data

            return HttpResponse(json.dumps(filtered_data), content_type='application/json')
        return wrapper
    return outwrap
