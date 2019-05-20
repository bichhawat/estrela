from rest_framework.views import APIView
from rest_framework.response import Response
from django.shortcuts import get_object_or_404
from v1.accounts.models.user import User
from rest_framework import status
from v1.accounts.serializers.user import UserSerializer

# follow/{user_id}
class FollowView(APIView):

    @staticmethod
    def post(request, user_id):
        """
        Follow a user
        """

        current_user = request.user
        to_follow = get_object_or_404(User, pk=user_id)

        current_user.following.add(to_follow)

        return Response(UserSerializer(to_follow).data, status=status.HTTP_201_CREATED)
