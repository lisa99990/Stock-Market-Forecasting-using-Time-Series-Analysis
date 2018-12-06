from django import forms
from django.conf import settings
import os

file_path = settings.BASE_DIR + '/files_system/'
filelist = [ f for f in os.listdir(file_path)  if f.endswith(".csv")]


class FileSelectForm(forms.Form):
	files = forms.ChoiceField(choices=[(f,f)for f in filelist],widget=forms.Select(attrs={'class':'form-control'}))