<?php

use Illuminate\Support\Facades\Route;

// Dev env equivalents:
//   client.api.php  → Route::post('/read_id_card', 'EmployeeController@readIdCard')      [auth: clients]
//   employee.api.php → Route::post('/read_card', 'EmployeeVerifyPersonalInfoController@readIdCard') [auth: employees]
//
// Middleware names below match the dev env convention. Adjust to whatever auth guards
// your mock Laravel project defines (e.g. auth:sanctum, auth:api, etc.).

/*
|--------------------------------------------------------------------------
| Client route — mirrors client.api.php line 124
| POST /api/client/read_id_card
|--------------------------------------------------------------------------
*/
Route::group([
    'prefix'     => 'client',
    'namespace'  => 'App\Http\Controllers\Client',
    'middleware' => ['api'],           // replace with 'auth.multi:clients' in real dev env
], function () {
    Route::post('/read_id_card', 'GeminiReadIdCardController@readIdCard');
});

/*
|--------------------------------------------------------------------------
| Employee route — mirrors employee.api.php line 926
| POST /api/employee/verify/read_card
|--------------------------------------------------------------------------
*/
Route::group([
    'prefix'     => 'employee',
    'namespace'  => 'App\Http\Controllers\Employee',
    'middleware' => ['api'],           // replace with 'auth.multi:employees' in real dev env
], function () {
    Route::group(['prefix' => 'verify'], function () {
        Route::post('/read_card', 'GeminiPersonalInfoController@readIdCard');
    });
});
