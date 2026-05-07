import React from 'react'
import ReactDOM from 'react-dom/client'
import { ConfigProvider } from 'antd'
import zhCN from 'antd/locale/zh_CN'
import App from './App'
import './theme.css'
import './index.css'

const theme = {
  token: {
    colorPrimary: '#1890ff',
    colorBgContainer: 'rgba(255,255,255,0.06)',
    colorText: 'rgba(255,255,255,0.9)',
    colorBorder: 'rgba(255,255,255,0.15)',
  },
  components: {
    Input: { activeBorderColor: '#1890ff', hoverBorderColor: '#40a9ff' },
    Button: { primaryShadow: '0 0 12px rgba(24,144,255,0.5)' },
  },
}

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <ConfigProvider locale={zhCN} theme={theme}>
      <App />
    </ConfigProvider>
  </React.StrictMode>,
)
