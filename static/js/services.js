app.service('mainDataService',[function(){
    
    this.sendData = function(id,dataObject){
        var key = firebase.database().ref('mainData/' + id).push().key;
        firebase.database().ref('mainData/' + id + '/' + key).set(dataObject).then(
            function(){
                console.log('Data added successfully');
                alert("Data added successfully");
            }
        )
    }
    
}])

