app.controller('lectureController',['$scope','mainDataService','$firebaseStorage','$firebaseAuth','$window','$firebaseArray',function($scope,mainDataService,$firebaseStorage,$firebaseAuth,$window,$firebaseArray){
    
    $scope.redirect = function(obj,cat){
        $window.location.href = "single.html?" + cat + "?" + obj.$id;
    }
    
    var authen = $firebaseAuth();
    
    authen.$onAuthStateChanged(function(firebaseUser){
        if(firebaseUser){
            //console.log(firebaseUser)
            $scope.username = firebaseUser.email;
            console.log($scope.username)
            $scope.loggedIn = true;
            console.log(firebaseUser);
            $scope.user = firebaseUser;
        }
    })
    
    $scope.logout = function(){
        authen.$signOut();
    }
    
    var ref = firebase.database().ref('mainData/lectures');
    var refObject = $firebaseArray(ref);
    
    $scope.myPromise = refObject.$loaded().then(
        function(data){
            $scope.list = data;
            console.log($scope.list)
        },
        function(){
            alert("An error occured");
        }
    )
    
}])

app.controller('workshopsController',['$scope','mainDataService','$firebaseStorage','$firebaseAuth','$window','$firebaseArray',function($scope,mainDataService,$firebaseStorage,$firebaseAuth,$window,$firebaseArray){
    
    $scope.redirect = function(obj,cat){
        $window.location.href = "single.html?" + cat + "?" + obj.$id;
    }
    
    var authen = $firebaseAuth();
    
    authen.$onAuthStateChanged(function(firebaseUser){
        if(firebaseUser){
            //console.log(firebaseUser)
            $scope.username = firebaseUser.email;
            console.log($scope.username)
            $scope.loggedIn = true;
            console.log(firebaseUser);
            $scope.user = firebaseUser;
        }
    })
    
    $scope.moreDetails = function(item){
        $window.open(item.pdf);
    }
    
    $scope.logout = function(){
        authen.$signOut();
    }
    
    var ref = firebase.database().ref('mainData/workshops');
    var refObject = $firebaseArray(ref);
    
    $scope.myPromise = refObject.$loaded().then(
        function(data){
            $scope.list = data;
            console.log($scope.list)
        },
        function(){
            alert("An error occured");
        }
    )
    
}])

app.controller('exhibitionsController',['$scope','mainDataService','$firebaseStorage','$firebaseAuth','$window','$firebaseArray',function($scope,mainDataService,$firebaseStorage,$firebaseAuth,$window,$firebaseArray){
    
    $scope.redirect = function(obj,cat){
        $window.location.href = "single.html?" + cat + "?" + obj.$id;
    }
    
    var authen = $firebaseAuth();
    
    authen.$onAuthStateChanged(function(firebaseUser){
        if(firebaseUser){
            //console.log(firebaseUser)
            $scope.username = firebaseUser.email;
            console.log($scope.username)
            $scope.loggedIn = true;
            console.log(firebaseUser);
            $scope.user = firebaseUser;
        }
    })
    
    $scope.moreDetails = function(item){
        $window.open(item.pdf);
    }
    
    $scope.logout = function(){
        authen.$signOut();
    }
    
    var ref = firebase.database().ref('mainData/exhibitions');
    var refObject = $firebaseArray(ref);
    
    $scope.myPromise = refObject.$loaded().then(
        function(data){
            $scope.list = data;
            console.log($scope.list)
        },
        function(){
            alert("An error occured");
        }
    )
    
}])

app.controller('registerController',['$scope','mainDataService','$firebaseStorage','$firebaseAuth','$window','$firebaseArray',function($scope,mainDataService,$firebaseStorage,$firebaseAuth,$window,$firebaseArray){
    
    $scope.formData = {
        email: "",
        password: "",
        password1: "",
        college: "",
        semester: "",
        branch: "",
        mob: "",
        name: "",
    }
    
    var authen = $firebaseAuth();
    
    authen.$onAuthStateChanged(function(firebaseUser){
        if(firebaseUser){
            //console.log(firebaseUser)
            $scope.username = firebaseUser.email;
            console.log($scope.username)
            $scope.loggedIn = true;
            $window.location.href = "index.html";
            console.log(firebaseUser);
            $scope.user = firebaseUser;
        }
    })
    
    $scope.sendData = function(){
        if($scope.formData.password != $scope.formData.password1){
            alert('Passwords do not match');
            return;
        }
        $scope.myPromise = authen.$createUserWithEmailAndPassword($scope.formData.email, $scope.formData.password)
          .then(function(firebaseUser) {
            console.log("User " + firebaseUser.uid + " created successfully!");
            firebase.database().ref('userList/' + firebaseUser.uid).set(
                        {
                email: $scope.formData.email,
                college: $scope.formData.college,
                semester: $scope.formData.semester,
                branch: $scope.formData.branch,
                mob: $scope.formData.mob,
                name: $scope.formData.name
            }
            ).then(
                function(){
                    alert('Registered successfully');
                    $window.location.href("login.html");
                },
                function(err){
                    
                    alert('An error occured');
                }
            )
          }).catch(function(error) {
            alert(error);
            console.error("Error: ", error);
          });
    }
    
    $scope.logout = function(){
        authen.$signOut();
    }
    
    var ref = firebase.database().ref('mainData/lectures');
    var refObject = $firebaseArray(ref);
    
    $scope.myPromise = refObject.$loaded().then(
        function(data){
            $scope.list = data;
            console.log($scope.list)
        },
        function(){
            alert("An error occured");
        }
    )
    
}])

app.controller('loginController',['$scope','mainDataService','$firebaseStorage','$firebaseAuth','$window','$firebaseArray','$location',function($scope,mainDataService,$firebaseStorage,$firebaseAuth,$window,$firebaseArray,$location){
    
    var path = $location.absUrl();
    var pathArray = path.split('?!');
    
    $scope.formData = {
        email: "",
        password: "",
        college: "",
        semester: "",
        branch: "",
        mob: ""
    }
    
    var authen = $firebaseAuth();
    
    
    
    authen.$onAuthStateChanged(function(firebaseUser){
        if(firebaseUser){
            //console.log(firebaseUser)
            $scope.username = firebaseUser.email;
            console.log($scope.username)
            $scope.loggedIn = true;
            if(!pathArray[1]){
                $window.location.href = "index.html";
            }
            else{
                $window.location.href = pathArray[1];
            }
            console.log(firebaseUser);
            $scope.user = firebaseUser;
        }
    })
    
    $scope.sendData = function(){
        $scope.myPromise = authen.$signInWithEmailAndPassword($scope.formData.email, $scope.formData.password).then(function(firebaseUser) {
              console.log("Signed in as:", firebaseUser.uid);
            }).catch(function(error) {
                alert(error);
              console.error("Authentication failed:", error);
        });
    }
    
    $scope.logout = function(){
        authen.$signOut();
    }
    
    var ref = firebase.database().ref('mainData/lectures');
    var refObject = $firebaseArray(ref);
    
    $scope.myPromise = refObject.$loaded().then(
        function(data){
            $scope.list = data;
            console.log($scope.list)
        },
        function(){
            alert("An error occured");
        }
    )
    
    
    
}])


app.controller('indexController',['$scope','mainDataService','$firebaseStorage','$firebaseAuth','$window','$firebaseObject',function($scope,mainDataService,$firebaseStorage,$firebaseAuth,$window,$firebaseObject){
    
    
    var authen = $firebaseAuth();
    
    authen.$onAuthStateChanged(function(firebaseUser){
        
       
        
        if(firebaseUser){
            
             var ref = firebase.database().ref('userList/' + firebaseUser.uid + '/name');
            var refObject = $firebaseObject(ref);
            $scope.myPromise = refObject.$loaded().then(
                function(data){
                    //$scope.username = data.$value;
                    console.log(data);
                }
            )
            //console.log(firebaseUser)
           
            console.log($scope.username)
            $scope.loggedIn = true;
            console.log(firebaseUser);
            $scope.user = firebaseUser;
            $scope.username = $scope.user.email;
        }
    })
    
    $scope.logout = function(){
        authen.$signOut();
    }
    
    
}])

app.controller('singleController',['$scope','mainDataService','$firebaseStorage','$firebaseAuth','$window','$firebaseObject','$location',function($scope,mainDataService,$firebaseStorage,$firebaseAuth,$window,$firebaseObject,$location){
    
    
    
    
    var authen = $firebaseAuth();
    
    authen.$onAuthStateChanged(function(firebaseUser){
        
        if(firebaseUser){
            
            var ref = firebase.database().ref('userList/' + firebaseUser.uid);
            var refObject = $firebaseObject(ref);
            $scope.myPromise = refObject.$loaded().then(
                function(data){
                    $scope.username = data.name;
                    $scope.userData = data;
                    console.log(data);
                }
            )
            //console.log(firebaseUser)
           
            console.log($scope.username)
            $scope.loggedIn = true;
            console.log(firebaseUser);
            $scope.user = firebaseUser;
        }
        else{
            var path1 = $location.absUrl();
            $window.location.href = "login.html?!" + path1;
        }
    })
    
    var path = $location.absUrl();
    console.log(path.split('?'));
    var pathArray = path.split('?');
    
    var ref1 = firebase.database().ref('mainData/' + pathArray[1] + '/' + pathArray[2]);
    var refObject1 = $firebaseObject(ref1);
    $scope.myPromise = refObject1.$loaded().then(
        function(data){
            console.log(data);
            $scope.datum = data;
        }
    )
    
    $scope.register = function(){

        $scope.myPromise = firebase.database().ref('registerData/' + pathArray[1] + '/' + pathArray[2] + '/' + $scope.user.uid).set(
            $scope.username
        ).then(
            function(){
                firebase.database().ref('userList/' + $scope.user.uid + '/' + pathArray[1] + '/' + pathArray[2]).set(
                    true
                ).then(
                    function(){
                        alert("Registered successfully. Please checkout your dashboard for more details.");
                    },
                    function(){
                        alert('An error occured')
                    }
                )
                
            }
        )
    }
    
    $scope.logout = function(){
        authen.$signOut();
        $window.location.href = "index.html";
    }
    
    
}])

