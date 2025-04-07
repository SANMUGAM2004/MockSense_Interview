from django import forms
from django.contrib.auth.models import User
from django.contrib.auth.password_validation import validate_password
from django.core.exceptions import ValidationError
import re

class RegisterForm(forms.ModelForm):
    password = forms.CharField(
        widget=forms.PasswordInput,
        label="Password"
    )
    confirm_password = forms.CharField(
        widget=forms.PasswordInput,
        label="Confirm Password"
    )

    class Meta:
        model = User
        fields = ['username', 'email', 'password']
        help_texts = {
            'username': None,
        }

    def clean_username(self):
            username = self.cleaned_data.get('username')

            # Length check
            if len(username) < 4 or len(username) > 20:
                raise forms.ValidationError("Username must be between 4 and 20 characters.")

            # Allowed characters: letters, numbers, _, @
            if not re.match(r'^[A-Za-z0-9_@]+$', username):
                raise forms.ValidationError("Username can only contain letters, numbers, underscores (_) and at symbols (@).")

            # Must include both lowercase and uppercase letters
            if not re.search(r'[a-z]', username):
                raise forms.ValidationError("Username must include at least one lowercase letter.")
            if not re.search(r'[A-Z]', username):
                raise forms.ValidationError("Username must include at least one uppercase letter.")

            # Uniqueness check
            if User.objects.filter(username=username).exists():
                raise forms.ValidationError("Username already taken.")

            return username

    def clean_email(self):
        email = self.cleaned_data.get('email')
        if User.objects.filter(email=email).exists():
            raise forms.ValidationError("Email is already registered.")
        return email

    def clean(self):
        cleaned_data = super().clean()
        password = cleaned_data.get("password")
        confirm = cleaned_data.get("confirm_password")
        if password and confirm:
            if password != confirm:
                raise forms.ValidationError("Passwords do not match.")
            try:
                validate_password(password)
            except ValidationError as e:
                self.add_error('password', e)

        return cleaned_data