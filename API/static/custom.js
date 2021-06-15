$(document).ready(function(){
    $("#link1").click(function(){
        console.log("link1")
        if(!$( "#link1" ).hasClass( "active" )){
            if($( "#link2" ).hasClass( "active" )){
                $( "#link2" ).removeClass("active")
                $( "#card2" ).addClass("d-none")
            }
            else{
                $( "#link3" ).removeClass("active")
                $( "#card3" ).addClass("d-none")
            }
            $( "#link1" ).addClass("active")
            $( "#card1" ).removeClass("d-none")
        }
    });

    $("#link2").click(function(){
        console.log("link2")
        if(!$( "#link2" ).hasClass( "active" )){
            if($( "#link1" ).hasClass( "active" )){
                $( "#link1" ).removeClass("active")
                $( "#card1" ).addClass("d-none")
            }
            else{
                $( "#link3" ).removeClass("active")
                $( "#card3" ).addClass("d-none")
            }
            $( "#link2" ).addClass("active")
            $( "#card2" ).removeClass("d-none")
        }
    });

    $("#link3").click(function(){
        console.log("link3")
        if(!$( "#link3" ).hasClass( "active" )){
            if($( "#link2" ).hasClass( "active" )){
                $( "#link2" ).removeClass("active")
                $( "#card2" ).addClass("d-none")
            }
            else{
                $( "#link1" ).removeClass("active")
                $( "#card1" ).addClass("d-none")
            }
            $( "#link3" ).addClass("active")
            $( "#card3" ).removeClass("d-none")
        }
    });

    $('#flexCheckDefault').change(function() {
        if(this.checked){ 
            $('#optional').removeClass("d-none")
            $('#customFile').prop('disabled', true)
        }
        else{
            $('#optional').addClass("d-none") 
            $('#customFile').prop('disabled', false) 
        }    
    });
});