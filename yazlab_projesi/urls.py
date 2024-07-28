from django.contrib import admin
from django.urls import path
from django.contrib.auth.views import LoginView, LogoutView
from yazlab3.views import signup, home, update_profile, article_detail
from yazlab3.veritabani import LoginForm
from yazlab3.besli import recommendations, update_interest_and_recommendations, replace_article, update_user_vector
from yazlab3 import besli
urlpatterns = [
    path('admin/', admin.site.urls),
    path('', home, name='home'),
    path('signup/', signup, name='signup'),
    path('login/', LoginView.as_view(template_name='login.html', authentication_form=LoginForm), name='login'),
    path('logout/', LogoutView.as_view(next_page='home'), name='logout'),
    path('recommendations/', recommendations, name='recommendations'),
    path('update-interest-and-recommendations/<str:article_id>/', update_interest_and_recommendations, name='update_interest_and_recommendations'),
    path('replace-article/<str:article_id>/<str:type>/', replace_article, name='replace_article'),
    path('update-profile/', update_profile, name='update_profile'),
    path('article/<str:article_id>/', article_detail, name='article_detail'),
    path('update-user-vector/<str:article_id>/', update_user_vector, name='update_user_vector'),
    path('search/', besli.search, name='search'),
]
