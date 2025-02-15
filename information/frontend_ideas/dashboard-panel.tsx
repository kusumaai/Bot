import React, { useState } from 'react';
import { LineChart, Line, AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { ArrowUpCircle, ArrowDownCircle, AlertCircle } from 'lucide-react';

const TradingDashboard = () => {
  const [balance, setBalance] = useState(10000);
  const [pnl, setPnl] = useState(2.5);
  const [activePositions, setActivePositions] = useState(2);
  const [alerts, setAlerts] = useState([
    { type: 'success', message: 'BTC/USDT Long +2.3%' },
    { type: 'error', message: 'ETH/USDT Stop Loss Hit' }
  ]);

  return (
    <div className="p-8 bg-gray-100 min-h-screen">
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
        <Card className="bg-white">
          <CardHeader>
            <CardTitle>Portfolio Balance</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold">
              ${balance.toLocaleString()}
            </div>
            <div className="text-green-500 flex items-center mt-2">
              <ArrowUpCircle className="w-4 h-4 mr-1" />
              {pnl}%
            </div>
          </CardContent>
        </Card>

        <Card className="bg-white">
          <CardHeader>
            <CardTitle>Active Positions</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold">{activePositions}</div>
            <div className="text-gray-500 mt-2">
              Max Positions: 5
            </div>
          </CardContent>
        </Card>

        <Card className="bg-white">
          <CardHeader>
            <CardTitle>System Health</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex items-center space-x-2">
              <div className="h-3 w-3 rounded-full bg-green-500"></div>
              <span>All Systems Operational</span>
            </div>
          </CardContent>
        </Card>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
        <Card className="bg-white">
          <CardHeader>
            <CardTitle>Equity Curve</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart
                  data={[
                    { time: '1', value: 10000 },
                    { time: '2', value: 10200 },
                    { time: '3', value: 10150 },
                    { time: '4', value: 10400 }
                  ]}
                >
                  <XAxis dataKey="time" />
                  <YAxis />
                  <Tooltip />
                  <Area type="monotone" dataKey="value" stroke="#8884d8" fill="#8884d8" fillOpacity={0.2} />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>

        <Card className="bg-white">
          <CardHeader>
            <CardTitle>Recent Alerts</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {alerts.map((alert, i) => (
                <Alert key={i} variant={alert.type === 'success' ? 'default' : 'destructive'}>
                  <AlertCircle className="h-4 w-4" />
                  <AlertDescription>{alert.message}</AlertDescription>
                </Alert>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default TradingDashboard;