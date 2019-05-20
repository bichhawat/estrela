from django.conf.urls import url
from .views.accept_invitation import AcceptInvitationView
# from .views.login import LoginView
# from .views.logout import LogoutView
from .views.profile import ProfileView, ProfileDetail
from .views.reset_password import ResetPasswordView
from .views.update_password import UpdatePasswordView
from .views.user import UserView, UserDetail
from .views.follow import FollowView

from django.contrib.auth import views as auth_views


urlpatterns = [

    # Accept invitation
    url(r'^accept_invitation$', AcceptInvitationView.as_view()),

    # Login / logout
    url(r'^login$', auth_views.LoginView.as_view()),
    url(r'^logout$', auth_views.LogoutView.as_view()),

    # Password management
    url(r'^reset_password$', ResetPasswordView.as_view()),
    url(r'^update_password$', UpdatePasswordView.as_view()),

    # Profiles
    url(r'^profiles$', ProfileView.as_view()),
    url(r'^profiles/(?P<profile_id>[\d]+)$', ProfileDetail.as_view()),

    # Users
    url(r'^users$', UserView.as_view()),
    url(r'^users/(?P<user_id>[\d]+)$', UserDetail.as_view()),

    # Follow User
    url(r'^follow/(?P<user_id>[\d]+)$', FollowView.as_view()),

]
