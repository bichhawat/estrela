## Steps

1. `cd /path/to/estrela-policy-server`

2. `cd server`

3. `virtualenv -p python3 env`

4. `source env/bin/activate`

5. `pip install -r requirements.txt`

6. `rm -rf env/lib/python3.6/site-packages/rest_framework/`

7. `cp -R ../no-rest-framework/ env/lib/python3.6/site-packages/rest_framework`

8. `python manage.py runserver`

Now in a different terminal

1. `cd /path/to/estrela-policy-server`

2. `cd estrela-server`

3. `virtualenv -p python3 env`

4. `source env/bin/activate`

5. `pip install -r requirements.txt`

6. `python manage.py runserver 8001`

You're done!!

