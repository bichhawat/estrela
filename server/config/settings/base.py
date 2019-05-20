import os
from .middleware import MIDDLEWARE

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SECRET_KEY = 'a1y425b-u7(1_j%ekoh+_v_d620p5l5zfl6+fq&vwd+suq^j*z'

INSTALLED_APPS = [

    # API (v1)
    'v1.accounts.apps.AccountsConfig',
    'v1.credits.apps.CreditsConfig',
    'v1.filters.apps.FiltersConfig',
    'v1.posts.apps.PostsConfig',
    'v1.private_messages.apps.PrivateMessagesConfig',
    'v1.replies.apps.RepliesConfig',
    'v1.user_roles.apps.UserRolesConfig',
    'v1.votes.apps.VotesConfig',

    # Base
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',

    # Requirements
    'corsheaders',
    'rest_framework',
    'rest_framework.authtoken',

    # Timelogging
    'timelog',
    'debug_toolbar',
]

STATIC_URL = '/static/'
    
ALLOWED_HOSTS = ['*']
AUTH_USER_MODEL = 'accounts.User'

ROOT_URLCONF = 'config.urls'
WSGI_APPLICATION = 'config.wsgi.application'

AUTH_PASSWORD_VALIDATORS = [
    {'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator', },
    {'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator', },
    {'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator', },
    {'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator', },
]

REST_FRAMEWORK = {
    'DEFAULT_PERMISSION_CLASSES': (
        'rest_framework.permissions.IsAuthenticatedOrReadOnly',
    ),
    'DEFAULT_AUTHENTICATION_CLASSES': (
        'rest_framework.authentication.BasicAuthentication',
        'rest_framework.authentication.TokenAuthentication',
        'rest_framework.authentication.SessionAuthentication',
    ),
}

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'UTC'
USE_I18N = True
USE_L10N = True
USE_TZ = True

FILE_UPLOAD_PERMISSIONS = 0o644

CORS_ORIGIN_ALLOW_ALL = True

CORS_ALLOW_HEADERS = [
    'accept',
    'accept-encoding',
    'authorization',
    'content-disposition',
    'content-type',
    'origin',
    'user-agent',
    'x-csrftoken',
    'x-requested-with',
]


# Logging
LOG_PATH = os.path.join(BASE_DIR, 'logs/')
TIMELOG_LOG = os.path.join(LOG_PATH, 'timelog.log')
SQL_LOG = os.path.join(LOG_PATH, 'sqllog.log')

LOGGING = {
    'version': 1,
    'formatters': {
        'plain': {
            'format': '%(asctime)s %(message)s'},
    },
    'handlers': {
        'timelog': {
            'level': 'DEBUG',
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': TIMELOG_LOG,
            'maxBytes': 1024 * 1024 * 5,  # 5 MB
            'backupCount': 5,
            'formatter': 'plain',
        },
        'sqllog': {
            'level': 'DEBUG',
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': SQL_LOG,
            'maxBytes': 1024 * 1024 * 5,  # 5 MB
            'backupCount': 5,
            'formatter': 'plain',
        },
    },

    'loggers': {
        'timelog.middleware': {
            'handlers': ['timelog'],
            'level': 'DEBUG',
            'propogate': False,
        },
        'timing_logging': {
            'handlers': ['timelog'],
            'level': 'DEBUG',
            'propogate': False,
        },
        'logging_middleware': {
            'handlers': ['sqllog'],
            'level': 'DEBUG',
            'propogate': False
        },
    }
}
