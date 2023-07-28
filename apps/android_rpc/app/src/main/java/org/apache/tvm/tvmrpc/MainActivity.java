/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.tvm.tvmrpc;

import android.content.Context;
import android.content.SharedPreferences;
import android.os.Bundle;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import android.widget.Button;
import android.widget.EditText;
import android.content.Intent;


public class MainActivity extends AppCompatActivity {
    @Nullable
    public static Integer tryParseInt(String text, Integer default_val) {
        try {
            return Integer.parseInt(text);
        } catch (NumberFormatException e) {
            return default_val;
        }
    }

    public Intent makeRPCIntent() {
        EditText edProxyAddress = findViewById(R.id.input_address);
        EditText edProxyPort = findViewById(R.id.input_port);
        EditText edAppKey = findViewById(R.id.input_key);
        EditText edCustomAddr = findViewById(R.id.input_custom_addr);

        final String proxyHost = edProxyAddress.getText().toString();
        final Integer proxyPort = tryParseInt(edProxyPort.getText().toString(), 9090);
        final String key = edAppKey.getText().toString();
        final String customAddr = edCustomAddr.getText().toString();

        SharedPreferences pref = getApplicationContext().getSharedPreferences("RPCProxyPreference", Context.MODE_PRIVATE);
        SharedPreferences.Editor editor = pref.edit();
        editor.putString("input_address", proxyHost);
        editor.putString("input_port", edProxyPort.getText().toString());
        editor.putString("input_key", key);
        editor.putString("input_custom_addr", customAddr);
        editor.commit();

        Intent intent = new Intent(this, RPCActivity.class);
        intent.putExtra("host", proxyHost);
        intent.putExtra("port", proxyPort);
        intent.putExtra("key", key);
        intent.putExtra("custom_addr", customAddr);
        return intent;
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        setSupportActionBar(findViewById(R.id.toolbar));
        Button start = findViewById(R.id.start_rpc);
        start.setOnClickListener(v -> startActivity(this.makeRPCIntent()));
    }
}
