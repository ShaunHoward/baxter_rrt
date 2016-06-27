from django.shortcuts import render_to_response, render


def test(request):
    return render(request, "test.html")
