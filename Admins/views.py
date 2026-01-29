from django.shortcuts import render, redirect
from django.contrib import messages
from Users.models import UserRegistrationModel

def AdminBase(request):
    return render(request, 'admins/AdminBase.html')

def AdminHome(request):
    return render(request, 'admins/AdminHome.html')

# Admin Login Check
def AdminLoginCheck(request):
    if request.method == 'POST':
        usrid = request.POST.get('loginid')
        pswd = request.POST.get('pswd')
        print("User ID is = ", usrid)
        if usrid == 'admin' and pswd == 'admin':
            return render(request, 'admins/AdminHome.html')
        else:
            messages.error(request, 'Please Check Your Login Details')
    return render(request, 'AdminLogin.html', {})

# View User Details
def UserDetails(request):
    user = UserRegistrationModel.objects.all()
    context = {'user': user}
    return render(request, 'admins/UserDetails.html', context)

# Activate Users
def ActivateUsers(request):
    if request.method == 'GET':
        id = request.GET.get('uid')
        status = 'activated'
        print("PID = ", id, status)
        UserRegistrationModel.objects.filter(id=id).update(status=status)
        user = UserRegistrationModel.objects.all()
        messages.success(request, 'User activated successfully')
        return render(request,'admins/UserDetails.html',{'user':user})
